from kdgan import config

from nets import nets_factory
from nets import vgg

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class DIS():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training

    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32, shape=(None, flags.feature_size))
    # None = batch_size * (num_positive + num_negative)
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    # None = batch_size * (num_positive + num_negative)
    self.label_ph = tf.placeholder(tf.float32, shape=(None,))

    dis_scope = 'discriminator'
    model_scope = nets_factory.arg_scopes_map[flags.model_name]
    with tf.variable_scope(dis_scope) as scope:
      with slim.arg_scope(model_scope(weight_decay=flags.dis_weight_decay)):
        net = self.image_ph
        net = slim.dropout(net, flags.dropout_keep_prob, 
            is_training=is_training)
        net = slim.fully_connected(net, config.num_label,
            activation_fn=None)
        self.logits = net

    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    self.rewards = 2 * (tf.sigmoid(sample_logits) - 0.5)

    if not is_training:
      return

    # save_dict = {}
    # for variable in tf.trainable_variables():
    #   if not variable.name.startswith(dis_scope):
    #     continue
    #   print('%s added to DIS saver' % variable.name)
    #   save_dict[variable.name] = variable
    # self.saver = tf.train.Saver(save_dict)

    global_step = tf.train.get_global_step()
    decay_steps = int(config.train_data_size / config.train_batch_size * flags.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(flags.init_learning_rate,
        global_step, decay_steps, flags.learning_rate_decay_factor,
        staircase=True, name='exponential_decay_learning_rate')

    tf.losses.sigmoid_cross_entropy(self.label_ph, sample_logits)
    losses = tf.get_collection(tf.GraphKeys.LOSSES)
    # print('#loss={}'.format(len(losses)))
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # print('#regularization_loss={}'.format(len(regularization_losses)))
    losses.extend(regularization_losses)
    loss = tf.add_n(losses, name='dis_loss')
    # total_loss = tf.losses.get_total_loss(name='total_loss')
    # diff = tf.subtract(loss, total_loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    self.train_op = optimizer.minimize(loss, global_step=global_step)

    # tf.summary.scalar('learning_rate', learning_rate)
    # tf.summary.scalar('loss', loss)
    # self.summary_op = tf.summary.merge_all()

