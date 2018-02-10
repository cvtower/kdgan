from kdgan import config
from kdgan import utils

import tensorflow as tf
from nets import nets_factory
from tensorflow.contrib import slim

class GEN():
  def __init__(self, flags, dataset, is_training=True, gen_scope='gen'):
    self.is_training = is_training
    
    # None = batch_size
    num_feature = flags.image_size * flags.image_size * flags.channels
    self.image_ph = tf.placeholder(tf.float32, shape=(None, num_feature))
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))
    self.soft_logit_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    self.gen_scope = gen_scope # = 'gen'
    with tf.variable_scope(gen_scope):
      self.logits = utils.get_logits(flags, 
          self.image_ph,
          flags.gen_model_name,
          flags.gen_weight_decay,
          flags.gen_keep_prob, 
          is_training=is_training)

      self.labels = tf.nn.softmax(self.logits)
      
      if not is_training:
        self.predictions = tf.argmax(self.logits, axis=1)
        self.accuracy = tf.equal(self.predictions, tf.argmax(self.hard_label_ph, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))
        return

      save_dict, var_list = {}, []
      for variable in tf.trainable_variables():
        if not variable.name.startswith(gen_scope):
          continue
        # print('%-50s added to GEN saver' % variable.name)
        save_dict[variable.name] = variable
        var_list.append(variable)
      self.saver = tf.train.Saver(save_dict)
      self.var_list = var_list

      self.global_step = tf.Variable(0, trainable=False)
      self.learning_rate = tf.Variable(flags.gen_learning_rate, trainable=False)
      # self.lr_update = tf.assign(self.learning_rate, self.learning_rate * flags.learning_rate_decay_factor)

      # pre train
      pre_losses = self.get_pre_losses(flags)
      self.pre_loss = tf.add_n(pre_losses, '%s_pre_loss' % gen_scope)
      pre_optimizer = utils.get_opt(flags, self.learning_rate)
      ## no clipping
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=self.global_step)
      # pre_grads_and_vars = pre_optimizer.compute_gradients(self.pre_loss, var_list)
      # pre_capped_grads_and_vars = [(gv[0], gv[1]) for gv in pre_grads_and_vars]
      # self.pre_update = pre_optimizer.apply_gradients(pre_capped_grads_and_vars, global_step=self.global_step)
      ## global clipping
      # pre_grads, pre_vars = zip(*pre_optimizer.compute_gradients(self.pre_loss, var_list))
      # pre_grads, _ = tf.clip_by_global_norm(pre_grads, flags.clip_norm)
      # self.pre_update = pre_optimizer.apply_gradients(zip(pre_grads, pre_vars), global_step=self.global_step)

      # kd train
      kd_losses = self.get_kd_losses(flags)
      self.kd_loss = tf.add_n(kd_losses, name='%s_kd_loss' % gen_scope)
      kd_optimizer = utils.get_opt(flags, self.learning_rate)
      self.kd_update = kd_optimizer.minimize(self.kd_loss, global_step=self.global_step)

      # gan train
      gan_losses = self.get_gan_losses(flags)
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % gen_scope)
      # gan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      gan_optimizer = utils.get_opt(flags, self.learning_rate)
      self.gan_update = gan_optimizer.minimize(self.gan_loss, global_step=self.global_step)


  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.gen_scope):
        continue
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_hard_loss(self):
    hard_loss = tf.losses.softmax_cross_entropy(self.hard_label_ph, self.logits)
    return hard_loss

  def get_pre_losses(self, flags):
    hard_loss = self.get_hard_loss()
    hard_loss *= 1.0 / flags.batch_size
    pre_losses = [hard_loss]
    print('#pre_losses=%d' % (len(pre_losses)))
    # pre_losses.extend(self.get_regularization_losses())
    # print('#pre_losses=%d' % (len(pre_losses)))
    return pre_losses

  def get_kd_losses(self, flags):
    hard_loss = self.get_hard_loss()
    # hard_loss = tf.losses.softmax_cross_entropy(self.hard_label_ph, gen_logits)
    # hard_loss *= flags.kd_hard_pct
    hard_loss *= 1.0 / flags.batch_size

    gen_logits = self.logits * (1.0 / flags.temperature)
    tch_logits = self.soft_logit_ph * (1.0 / flags.temperature)

    if flags.kd_model == 'mimic':
      soft_loss = tf.losses.mean_squared_error(tch_logits, gen_logits)
    elif flags.kd_model == 'distn':
      soft_loss = tf.losses.softmax_cross_entropy(tch_logits, gen_logits)
      soft_loss *= pow(flags.temperature, 2.0)
    elif flags.kd_model == 'noisy':
      soft_loss = 0.0
    else:
      raise ValueError('bad kd model %s', flags.kd_model)
    soft_loss *= flags.kd_soft_pct / flags.batch_size

    kd_losses = [hard_loss, soft_loss]
    print('#kd_losses=%d' % (len(kd_losses)))
    # kd_losses.extend(self.get_regularization_losses())
    # print('#kd_losses=%d' % (len(kd_losses)))
    return kd_losses

  def get_gan_losses(self, flags):
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    # gan_loss = -tf.reduce_mean(self.reward_ph * sample_logits)
    gan_losses = [tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)]
    gan_losses.extend(self.get_regularization_losses())
    return gan_losses















