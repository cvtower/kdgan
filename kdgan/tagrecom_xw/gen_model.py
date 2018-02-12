from kdgan import config, utils

from nets import nets_factory
from nets import vgg

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class GEN():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32, shape=(None, flags.feature_size))
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))
    self.soft_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    self.gen_scope = gen_scope = 'gen'
    model_scope = nets_factory.arg_scopes_map[flags.gen_model_name]
    with tf.variable_scope(gen_scope) as scope:
      with slim.arg_scope(model_scope(weight_decay=flags.gen_weight_decay)):
        net = self.image_ph
        net = slim.dropout(net, flags.gen_keep_prob, is_training=is_training)
        net = slim.fully_connected(net, flags.num_label, activation_fn=None)
        self.logits = net

      self.labels = tf.nn.softmax(self.logits)

      if not is_training:
        return

      save_dict = {}
      for variable in tf.trainable_variables():
        if not variable.name.startswith(gen_scope):
          continue
        print('%-50s added to GEN saver' % variable.name)
        save_dict[variable.name] = variable
      self.saver = tf.train.Saver(save_dict)

      global_step = tf.Variable(0, trainable=False)
      tn_size = utils.get_tn_size(flags.dataset)
      self.learning_rate = utils.get_lr(flags, 
          tn_size,
          global_step,
          flags.gen_learning_rate,
          gen_scope)

      # pre train
      pre_losses = self.get_pre_losses(flags)
      self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % gen_scope)
      pre_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=global_step)

      # kd train
      kd_losses = self.get_kd_losses(flags)
      self.kd_loss = tf.add_n(kd_losses, name='%s_kd_loss' % gen_scope)
      kd_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.kd_update = kd_optimizer.minimize(self.kd_loss, global_step=global_step)

      # gan train
      gan_losses = self.get_gan_losses(flags)
      gan_losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % gen_scope)
      gan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.gan_update = gan_optimizer.minimize(self.gan_loss, global_step=global_step)

      # kdgan train
      kdgan_losses = self.get_kd_losses(flags) + self.get_gan_losses(flags)
      self.kdgan_loss = tf.add_n(kdgan_losses, name='%s_kdgan_loss' % gen_scope)
      kdgan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.kdgan_update = kdgan_optimizer.minimize(self.kdgan_loss, global_step=global_step)

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      # print('gen model:%s' % (regularization_loss.name))
      if not regularization_loss.name.startswith(self.gen_scope):
        continue
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_pre_losses(self, flags):
    pre_losses = [tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits)]
    print('#pre_losses wo regularization=%d' % (len(pre_losses)))
    pre_losses.extend(self.get_regularization_losses())
    print('#pre_losses wt regularization=%d' % (len(pre_losses)))
    return pre_losses

  def get_kd_losses(self, flags):
    hard_loss = tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits)
    hard_loss *= (1.0 - flags.kd_soft_pct)

    smooth_labels = tf.nn.softmax(self.soft_label_ph / flags.temperature)
    soft_loss = tf.nn.l2_loss(tf.nn.softmax(self.logits) - smooth_labels)
    soft_loss *= flags.kd_soft_pct

    kd_losses = [hard_loss, soft_loss]
    return kd_losses

  def get_gan_losses(self, flags):
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    # gan_loss = -tf.reduce_mean(self.reward_ph * sample_logits)
    gan_losses = [tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)]
    return gan_losses

