from kdgan import config

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class TCH():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training

    self.text_ph = tf.placeholder(tf.int64, shape=(None, None))
    self.label_ph = tf.placeholder(tf.float32, shape=(None, config.num_label))

    tch_scope = 'teacher'
    initializer = tf.random_uniform([config.vocab_size, flags.embedding_size], -0.1, 0.1)
    with tf.variable_scope(tch_scope) as scope:
      word_embedding = tf.get_variable('word_embedding', initializer=initializer)
      text_embedding = tf.nn.embedding_lookup(word_embedding, self.text_ph)
      text_embedding = tf.reduce_mean(text_embedding, axis=-2)
      self.logits = slim.fully_connected(text_embedding, config.num_label,
                activation_fn=None)

    if not is_training:
      return

    save_dict = {}
    for variable in tf.trainable_variables():
      if not variable.name.startswith(tch_scope):
        continue
      print('add %s to tch save dict' % variable.name)
      save_dict[variable.name] = variable
    self.saver = tf.train.Saver(save_dict)

    global_step = tf.train.get_global_step()
    decay_steps = int(config.train_data_size / config.train_batch_size * flags.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(flags.init_learning_rate,
        global_step, decay_steps, flags.learning_rate_decay_factor,
        staircase=True, name='exponential_decay_learning_rate')
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=self.label_ph, logits=self.logits))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    self.train_op = optimizer.minimize(loss, global_step=global_step)

    tf.summary.scalar('loss', loss)
    self.summary_op = tf.summary.merge_all()

