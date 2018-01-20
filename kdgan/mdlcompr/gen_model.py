from kdgan import config
from kdgan import utils

import tensorflow as tf
from nets import nets_factory
from tensorflow.contrib import slim

class GEN():
  def __init__(self, flags, dataset, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    num_feature = flags.image_size * flags.image_size * flags.channels
    self.image_ph = tf.placeholder(tf.float32, shape=(None, num_feature))
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))
    self.soft_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    hidden_size = 800
    self.gen_scope = gen_scope = 'gen'
    with tf.variable_scope(gen_scope):
      self.logits = utils.build_mlp_logits(flags, self.image_ph,
        hidden_size=hidden_size,
        keep_prob=flags.gen_keep_prob,
        weight_decay=flags.gen_weight_decay,
        is_training=is_training)

      self.labels = tf.nn.softmax(self.logits)
      
      if not is_training:
        self.predictions = tf.argmax(self.logits, axis=1)
        return

      save_dict, var_list = {}, []
      for variable in tf.trainable_variables():
        if not variable.name.startswith(gen_scope):
          continue
        print('%-50s added to GEN saver' % variable.name)
        save_dict[variable.name] = variable
        var_list.append(variable)
      self.saver = tf.train.Saver(save_dict)

      self.global_step = tf.Variable(0, trainable=False)
      # self.learning_rate = utils.get_lr(flags, self.global_step, dataset.num_examples, gen_scope)
      self.learning_rate = tf.Variable(flags.learning_rate, trainable=False)
      self.lr_update = tf.assign(self.learning_rate, self.learning_rate / 2.0)

      # pre train
      pre_losses = self.get_pre_losses()
      self.pre_loss = tf.add_n(pre_losses, '%s_pre_loss' % gen_scope)
      pre_optimizer = utils.get_opt(flags, self.learning_rate, opt_epsilon=flags.gen_opt_epsilon)
      ## no clipping
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=self.global_step)
      # pre_grads_and_vars = pre_optimizer.compute_gradients(self.pre_loss, var_list)
      # pre_capped_grads_and_vars = [(gv[0], gv[1]) for gv in pre_grads_and_vars]
      # self.pre_update = pre_optimizer.apply_gradients(pre_capped_grads_and_vars, global_step=self.global_step)
      ## global clipping
      # pre_grads, pre_vars = zip(*pre_optimizer.compute_gradients(self.pre_loss, var_list))
      # pre_grads, _ = tf.clip_by_global_norm(pre_grads, flags.clip_norm)
      # self.pre_update = pre_optimizer.apply_gradients(zip(pre_grads, pre_vars), global_step=self.global_step)

      # gan train
      gan_losses = self.get_gan_losses(flags)
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % gen_scope)
      gan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.gan_update = gan_optimizer.minimize(self.gan_loss, global_step=self.global_step)


  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.gen_scope):
        continue
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_pre_losses(self):
    pre_losses = [tf.losses.softmax_cross_entropy(self.hard_label_ph, self.logits)]
    print('#pre_losses=%d' % (len(pre_losses)))
    pre_losses.extend(self.get_regularization_losses())
    print('#pre_losses=%d' % (len(pre_losses)))
    return pre_losses

  def get_gan_losses(self, flags):
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    # gan_loss = -tf.reduce_mean(self.reward_ph * sample_logits)
    gan_losses = [tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)]
    gan_losses.extend(self.get_regularization_losses())
    return gan_losses















