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
    self.hard_label_ph = tf.placeholder(tf.int32, shape=(None,))

    self.gen_scope = gen_scope = 'gen'
    hidden_size = 800
    with tf.variable_scope(gen_scope):
      with tf.variable_scope('fc1'):
        fc1_weights = tf.get_variable('weights', [num_feature, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer())
        fc1_biases = tf.get_variable('biases', [hidden_size],
            initializer=tf.zeros_initializer())

      with tf.variable_scope('fc2'):
        fc2_weights = tf.get_variable('weights', [hidden_size, hidden_size],
            initializer=tf.contrib.layers.xavier_initializer())
        fc2_biases = tf.get_variable('biases', [hidden_size],
            initializer=tf.zeros_initializer())

      with tf.variable_scope('fc3'):
        fc3_weights = tf.get_variable('weights',[hidden_size, flags.num_label],
            initializer=tf.contrib.layers.xavier_initializer())
        fc3_biases = tf.get_variable('biases', [flags.num_label],
            initializer=tf.zeros_initializer())

      fc1 = tf.add(tf.matmul(self.image_ph, fc1_weights), fc1_biases)
      fc1 = tf.nn.relu(fc1)
      fc1 = tf.layers.dropout(fc1,
          keep_prob=flags.dropout_keep_prob,
          is_training=is_training)

      fc2 = tf.add(tf.matmul(fc1, fc2_weights), fc2_biases)
      fc2 = tf.nn.relu(fc2)
      fc2 = tf.layers.dropout(fc2,
          keep_prob=flags.dropout_keep_prob,
          is_training=is_training)

      self.logits = tf.matmul(fc2, fc3_weights) + fc3_biases

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
      self.learning_rate = utils.get_lr(flags, self.global_step, dataset.num_examples, gen_scope)

      encoded_labels = slim.one_hot_encoding(self.hard_label_ph, flags.num_label)
      tf.losses.softmax_cross_entropy(encoded_labels, self.logits)
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





















