from kdgan import config
from kdgan import utils

import numpy as np
import tensorflow as tf
from nets import nets_factory
from tensorflow.contrib import slim

class STD():
  def __init__(self, flags, is_training=True, std_scope='std'):
    self.is_training = is_training
    
    # None = batch_size
    image_shape = (None, flags.image_size, flags.image_size, flags.channels)
    self.image_ph = tf.placeholder(tf.float32, shape=image_shape)
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))
    self.soft_logit_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    self.std_scope = std_scope # = 'std'
    with tf.variable_scope(std_scope):
      network_fn = nets_factory.get_network_fn(flags.std_model_name,
          num_classes=flags.num_label,
          weight_decay=flags.std_weight_decay,
          is_training=is_training)
      self.logits, _ = network_fn(self.image_ph, dropout_keep_prob=flags.std_keep_prob)
      self.labels = tf.nn.softmax(self.logits)
      
      if not is_training:
        self.predictions = tf.argmax(self.logits, axis=1)
        self.accuracy = tf.equal(self.predictions, tf.argmax(self.hard_label_ph, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))
        return

      save_dict, var_list = {}, []
      for variable in tf.trainable_variables():
        if not variable.name.startswith(std_scope):
          continue
        print('%-50s added to STD saver' % variable.name)
        save_dict[variable.name] = variable
        var_list.append(variable)
      self.saver = tf.train.Saver(save_dict)
      self.var_list = var_list

      self.global_step = tf.Variable(0, trainable=False)
      self.learning_rate = tf.Variable(flags.std_learning_rate, trainable=False)
      # self.lr_update = tf.assign(self.learning_rate, self.learning_rate * flags.learning_rate_decay_factor)

      # pre train
      pre_losses = self.get_pre_losses(flags)
      pre_losses.extend(self.get_regularization_losses())
      print('#pre_losses wt regularization=%d' % (len(pre_losses)))
      self.pre_loss = tf.add_n(pre_losses, '%s_pre_loss' % std_scope)
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
      self.kd_loss = tf.add_n(kd_losses, name='%s_kd_loss' % std_scope)
      kd_optimizer = utils.get_opt(flags, self.learning_rate)
      self.kd_update = kd_optimizer.minimize(self.kd_loss, global_step=self.global_step)

      # gan train
      gan_losses = self.get_gan_losses(flags)
      gan_losses.extend(self.get_regularization_losses())
      print('#gan_losses wt regularization=%d' % (len(gan_losses)))
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % std_scope)
      gan_optimizer = utils.get_opt(flags, self.learning_rate)
      # gan_optimizer = tf.train.AdamOptimizer(self.learning_rate)
      # gan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.gan_update = gan_optimizer.minimize(self.gan_loss, global_step=self.global_step)

      # kdgan train
      kdgan_losses = self.get_kdgan_losses(flags)
      kdgan_losses.extend(self.get_regularization_losses())
      print('#kdgan_losses wt regularization=%d' % (len(kdgan_losses)))
      self.kdgan_loss = tf.add_n(kdgan_losses, name='%s_kdgan_loss' % std_scope)
      kdgan_optimizer = utils.get_opt(flags, self.learning_rate)
      self.kdgan_update = kdgan_optimizer.minimize(self.kdgan_loss, global_step=self.global_step)

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.std_scope):
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
    # print('#pre_losses wo regularization=%d' % (len(pre_losses)))
    return pre_losses

  def get_kd_losses(self, flags):
    kd_losses = []
    if flags.kd_model == 'mimic':
      soft_loss = tf.nn.l2_loss(self.soft_logit_ph - self.logits) / flags.batch_size
      kd_losses.append(soft_loss)
    elif flags.kd_model == 'distn':
      hard_loss = self.get_hard_loss()
      hard_loss *= (1.0 - flags.kd_soft_pct) / flags.batch_size
      std_logits = self.logits * (1.0 / flags.temperature)
      tch_logits = self.soft_logit_ph * (1.0 / flags.temperature)
      soft_loss = tf.losses.mean_squared_error(tch_logits, std_logits)
      soft_loss *= (pow(flags.temperature, 2.0) * flags.kd_soft_pct) / flags.batch_size
      kd_losses.extend([hard_loss, soft_loss])
    elif flags.kd_model == 'noisy':
      # self.noisy = noisy = tf.multiply(noisy_mask, gaussian)
      # tch_logits = tf.multiply(self.soft_logit_ph, tf.tile(noisy, tf.constant([1, flags.num_label])))
      # soft_loss = tf.nn.l2_loss(tch_logits - self.logits) / flags.batch_size
      # kd_losses.append(soft_loss)
      noisy = np.float32(np.ones((flags.batch_size, flags.num_label)))
      noisy = tf.nn.dropout(noisy, keep_prob=(1.0 - flags.noisy_ratio))
      noisy += tf.random_normal((flags.batch_size, flags.num_label), stddev=flags.noisy_sigma)
      tch_logits = tf.multiply(self.soft_logit_ph, noisy)
      soft_loss = tf.nn.l2_loss(tch_logits - self.logits) / flags.batch_size
      kd_losses.append(soft_loss)
    else:
      raise ValueError('bad kd model %s', flags.kd_model)
    # print('#kd_losses wo regularization=%d' % (len(kd_losses)))
    return kd_losses

  def get_gan_losses(self, flags):
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    gan_loss = tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)
    gan_losses = [gan_loss]
    # print('#gan_losses wo regularization=%d' % (len(gan_losses)))
    return gan_losses

  def get_kdgan_losses(self, flags):
    kdgan_losses = []
    for gan_loss in self.get_gan_losses(flags):
      gan_loss *= (1.0 - flags.intelltch_weight)
      kdgan_losses.append(gan_loss)
    for kd_loss in self.get_kd_losses(flags):
      kd_loss *= flags.distilled_weight
      kdgan_losses.append(kd_loss)
    # print('#kdgan_losses wo regularization=%d' % (len(kdgan_losses)))
    return kdgan_losses













