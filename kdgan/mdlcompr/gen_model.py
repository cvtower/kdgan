from kdgan import config
from kdgan import utils

import tensorflow as tf
from nets import nets_factory
from tensorflow.contrib import slim

class GEN():
  def __init__(self, flags, dataset, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32,
        shape=(None, flags.image_size, flags.image_size, flags.channels))
    self.hard_label_ph = tf.placeholder(tf.int32, shape=(None,))

    self.gen_scope = gen_scope = 'gen'
    hidden_size = 512
    with tf.variable_scope(gen_scope) as scope:
      with slim.arg_scope(self.gen_arg_scope(weight_decay=flags.weight_decay)):
        net = slim.flatten(self.image_ph)
        net = slim.fully_connected(net, hidden_size, scope='fc1')
        
        net = slim.dropout(net, flags.dropout_keep_prob,
            is_training=is_training,
            scope='dropout1')
        net = slim.fully_connected(net, hidden_size, scope='fc2')

        net = slim.dropout(net, flags.dropout_keep_prob,
            is_training=is_training,
            scope='dropout2')
        self.logits = slim.fully_connected(net, dataset.num_classes, 
            activation_fn=None, scope='fc3')

        if not is_training:
          self.predictions = tf.argmax(self.logits, axis=1)
          return

        save_dict = {}
        for variable in tf.trainable_variables():
          if not variable.name.startswith(gen_scope):
            continue
          print('%-50s added to GEN saver' % variable.name)
          save_dict[variable.name] = variable
        self.saver = tf.train.Saver(save_dict)

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = utils.get_lr(flags, self.global_step, dataset.num_samples, gen_scope)

        encoded_labels = slim.one_hot_encoding(self.hard_label_ph, dataset.num_classes)
        tf.losses.sigmoid_cross_entropy(encoded_labels, self.logits)
        pre_losses = []
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
          if not loss.name.startswith(gen_scope):
            continue
          print('%-50s added to GEN loss' % loss.name)
          pre_losses.append(loss)
        print('#loss=%d' % (len(pre_losses)))
        for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
          if not regularization_loss.name.startswith(gen_scope):
            continue
          print('%-50s added to GEN loss' % regularization_loss.name)
          pre_losses.append(regularization_loss)
        print('#loss=%d' % (len(pre_losses)))
        self.pre_loss = tf.add_n(pre_losses, '%s_pre_loss' % gen_scope)
        pre_optimizer = utils.get_opt(flags, self.learning_rate)
        self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=self.global_step)

  def gen_arg_scope(self, weight_decay=0.0):
    with slim.arg_scope(
        [slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        activation_fn=tf.nn.relu) as sc:
      return sc

