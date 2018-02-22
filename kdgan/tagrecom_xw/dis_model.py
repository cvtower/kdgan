from kdgan import config
from kdgan import utils

import tensorflow as tf
from tensorflow.contrib import slim

class DIS():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training

    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32, shape=(None, flags.feature_size))
    self.text_ph = tf.placeholder(tf.int64, shape=(None, None))
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))
    self.soft_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,))

    self.dis_scope = dis_scope = 'dis'
    vocab_size = utils.get_vocab_size(flags.dataset)
    # initializer = tf.random_uniform([vocab_size, flags.embedding_size], -0.1, 0.1)
    with tf.variable_scope(dis_scope) as scope:
      with slim.arg_scope([slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(flags.image_weight_decay)):
        image_embedding = self.image_ph
        image_embedding = slim.dropout(image_embedding, flags.dis_keep_prob,
            is_training=is_training)

      with slim.arg_scope([slim.fully_connected],
          weights_regularizer=slim.l2_regularizer(flags.text_weight_decay)):
        word_embedding = slim.variable('word_embedding',
            shape=[vocab_size, flags.embedding_size],
            # regularizer=slim.l2_regularizer(flags.text_weight_decay),
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        # word_embedding = tf.get_variable('word_embedding', initializer=initializer)
        text_embedding = tf.nn.embedding_lookup(word_embedding, self.text_ph)
        text_embedding = tf.reduce_mean(text_embedding, axis=-2)

      with slim.arg_scope([slim.fully_connected],
          activation_fn=tf.nn.relu,
          weights_regularizer=slim.l2_regularizer(flags.dis_weight_decay),
          biases_initializer=tf.zeros_initializer()):
        # combined_layer = tf.concat([text_embedding], 1)
        combined_layer = tf.concat([image_embedding, text_embedding], 1)
        self.logits = slim.fully_connected(combined_layer, flags.num_label,
            activation_fn=None)

      self.labels = tf.nn.softmax(self.logits)

      if not is_training:
        return

      save_dict = {}
      for variable in tf.trainable_variables():
        if not variable.name.startswith(dis_scope):
          continue
        print('%-50s added to DIS saver' % variable.name)
        save_dict[variable.name] = variable
      self.saver = tf.train.Saver(save_dict)

      self.global_step = global_step = tf.Variable(0, trainable=False)
      tn_size = utils.get_tn_size(flags.dataset)
      self.learning_rate = utils.get_lr(
          flags,
          tn_size,
          self.global_step,
          flags.dis_learning_rate,
          dis_scope)

      # pre train
      pre_losses = self.get_pre_losses()
      self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % dis_scope)
      pre_optimizer = utils.get_opt(flags, self.learning_rate)
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=global_step)

      # kdgan train
      sample_logits = tf.gather_nd(self.logits, self.sample_ph)
      kdgan_losses = [tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)]
      self.kdgan_loss = tf.add_n(kdgan_losses, name='%s_kdgan_loss' % dis_scope)
      kdgan_optimizer = utils.get_opt(flags, self.learning_rate)
      self.kdgan_update = kdgan_optimizer.minimize(self.kdgan_loss, global_step=global_step)

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      # print('gen model:%s' % (regularization_loss.name))
      if not regularization_loss.name.startswith(self.dis_scope):
        continue
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_pre_losses(self):
    pre_losses = [tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits)]
    print('%s #pre_losses wo regularization=%d' % (self.dis_scope, len(pre_losses)))
    pre_losses.extend(self.get_regularization_losses())
    print('%s #pre_losses wt regularization=%d' % (self.dis_scope, len(pre_losses)))
    return pre_losses
