from kdgan import config
from kdgan import utils

import tensorflow as tf
from nets import nets_factory
from tensorflow.contrib import slim

class TCH():
  def __init__(self, flags, dataset, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    num_feature = flags.image_size * flags.image_size * flags.channels
    self.image_ph = tf.placeholder(tf.float32, shape=(None, num_feature))
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))
    self.soft_logit_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    self.tch_scope = tch_scope = 'tch'
    with tf.variable_scope(tch_scope) as scope:
      self.logits = utils.get_logits(flags, 
          self.image_ph,
          flags.tch_model_name,
          flags.tch_weight_decay,
          flags.tch_keep_prob, 
          is_training=is_training)

      self.labels = tf.nn.softmax(self.logits)

      if not is_training:
        self.predictions = tf.argmax(self.logits, axis=1)
        self.accuracy = tf.equal(self.predictions, tf.argmax(self.hard_label_ph, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))
        return

      save_dict = {}
      for variable in tf.trainable_variables():
        if not variable.name.startswith(tch_scope):
          continue
        print('%-50s added to TCH saver' % variable.name)
        save_dict[variable.name] = variable
      self.saver = tf.train.Saver(save_dict)

      self.global_step = tf.Variable(0, trainable=False)
      self.learning_rate = tf.Variable(flags.gen_learning_rate, trainable=False)
      # self.lr_update = tf.assign(self.learning_rate, self.learning_rate * flags.learning_rate_decay_factor)

      pre_losses = self.get_pre_losses()
      self.pre_loss = tf.add_n(pre_losses, '%s_pre_loss' % tch_scope)
      pre_optimizer = utils.get_opt(flags, self.learning_rate)
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=self.global_step)

      # kdgan train
      kdgan_losses = self.get_kdgan_losses(flags)
      self.kdgan_loss = tf.add_n(kdgan_losses, name='%s_kdgan_loss' % tch_scope)
      kdgan_optimizer = utils.get_opt(flags, self.learning_rate)
      self.kdgan_update = kdgan_optimizer.minimize(self.kdgan_loss, global_step=self.global_step)

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.tch_scope):
        continue
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_pre_losses(self):
    pre_losses = [tf.losses.softmax_cross_entropy(self.hard_label_ph, self.logits)]
    print('#pre_losses=%d' % (len(pre_losses)))
    pre_losses.extend(self.get_regularization_losses())
    print('#pre_losses=%d' % (len(pre_losses)))
    return pre_losses

  def get_kdgan_losses(self, flags):
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    # kdgan_losses = -tf.reduce_mean(self.reward_ph * sample_logits)
    kdgan_losses = [tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)]
    kdgan_losses.extend(self.get_regularization_losses())
    return kdgan_losses














