from kdgan import config
from kdgan import utils

import tensorflow as tf
from nets import nets_factory
from tensorflow.contrib import slim


n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

with tf.name_scope('weight'):
    normal_weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name='w1_normal'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]),name='w2_normal'),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]),name='wout_normal')
    }
    truncated_normal_weights  = {
        'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],stddev=0.1),name='w1_truncated_normal'),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],stddev=0.1),name='w2_truncated_normal'),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, n_classes],stddev=0.1),name='wout_truncated_normal')
    }
    xavier_weights  = {
        'h1': tf.get_variable('w1_xaiver', [n_input, n_hidden_1],initializer=tf.contrib.layers.xavier_initializer()),
        'h2': tf.get_variable('w2_xaiver', [n_hidden_1, n_hidden_2],initializer=tf.contrib.layers.xavier_initializer()),
        'out': tf.get_variable('wout_xaiver',[n_hidden_2, n_classes],initializer=tf.contrib.layers.xavier_initializer())
    }
    he_weights = {
        'h1': tf.get_variable('w1_he', [n_input, n_hidden_1],
                              initializer=tf.contrib.layers.variance_scaling_initializer()),
        'h2': tf.get_variable('w2_he', [n_hidden_1, n_hidden_2],
                              initializer=tf.contrib.layers.variance_scaling_initializer()),
        'out': tf.get_variable('wout_he', [n_hidden_2, n_classes],
                               initializer=tf.contrib.layers.variance_scaling_initializer())
    }
with tf.name_scope('bias'):
    normal_biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1]),name='b1_normal'),
        'b2': tf.Variable(tf.random_normal([n_hidden_2]),name='b2_normal'),
        'out': tf.Variable(tf.random_normal([n_classes]),name='bout_normal')
    }
    zero_biases = {
        'b1': tf.Variable(tf.zeros([n_hidden_1]),name='b1_zero'),
        'b2': tf.Variable(tf.zeros([n_hidden_2]),name='b2_zero'),
        'out': tf.Variable(tf.zeros([n_classes]),name='bout_normal')
    }
weight_initializer = {'normal':normal_weights, 'truncated_normal':truncated_normal_weights, 'xavier':xavier_weights, 'he':he_weights}
bias_initializer = {'normal':normal_biases, 'zero':zero_biases}

# Create model of MLP without batch-normalization layer
def MLPwoBN(x, weights, biases):
    with tf.name_scope('MLPwoBN'):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

class GEN():
  def __init__(self, flags, dataset, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    feature_size = flags.image_size * flags.image_size * flags.channels
    self.image_ph = tf.placeholder(tf.float32, shape=(None, feature_size))
    self.hard_label_ph = tf.placeholder(tf.int32, shape=(None,))

    self.gen_scope = gen_scope = 'gen'
    hidden_size = 256
    with tf.variable_scope(gen_scope) as scope:
      with slim.arg_scope(self.gen_arg_scope(weight_decay=flags.weight_decay)):
        # net = slim.flatten(self.image_ph)
        # net = slim.fully_connected(net, hidden_size, scope='fc1')
        
        # net = slim.dropout(net, flags.dropout_keep_prob,
        #     is_training=is_training,
        #     scope='dropout1')
        # net = slim.fully_connected(net, hidden_size, scope='fc2')

        # net = slim.dropout(net, flags.dropout_keep_prob,
        #     is_training=is_training,
        #     scope='dropout2')
        # self.logits = slim.fully_connected(net, flags.num_label, 
        #     activation_fn=None, scope='fc3')

        weights = weight_initializer['xavier']
        biases = bias_initializer['zero']
        self.logits = MLPwoBN(self.image_ph, weights, biases)

        if not is_training:
          self.predictions = tf.argmax(self.logits, axis=1)
          return

        # save_dict = {}
        # for variable in tf.trainable_variables():
        #   if not variable.name.startswith(gen_scope):
        #     continue
        #   print('%-50s added to GEN saver' % variable.name)
        #   save_dict[variable.name] = variable
        # self.saver = tf.train.Saver(save_dict)

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

  def gen_arg_scope(self, weight_decay=0.0):
    with slim.arg_scope(
        [slim.fully_connected],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        # weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.zeros_initializer(),
        activation_fn=tf.nn.relu) as sc:
      return sc

