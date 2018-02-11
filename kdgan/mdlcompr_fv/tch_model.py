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

    self.tch_scope = tch_scope = 'tch'
    with tf.variable_scope(tch_scope) as scope:
      network_fn = nets_factory.get_network_fn(flags.tch_model_name,
          num_classes=flags.num_label,
          weight_decay=flags.tch_weight_decay,
          is_training=is_training)
      assert flags.image_size==network_fn.default_image_size
      net = tf.reshape(self.image_ph, [-1, flags.image_size, flags.image_size, flags.channels])
      self.logits, _ = network_fn(net, dropout_keep_prob=flags.tch_keep_prob)

      if not is_training:
        self.predictions = tf.argmax(self.logits, axis=1)
        return

      save_dict = {}
      for variable in tf.trainable_variables():
        if not variable.name.startswith(tch_scope):
          continue
        print('%-50s added to TCH saver' % variable.name)
        save_dict[variable.name] = variable
      self.saver = tf.train.Saver(save_dict)

      self.global_step = tf.Variable(0, trainable=False)
      self.learning_rate = utils.get_lr(flags, 
          self.global_step,
          dataset.num_examples,
          flags.tch_learning_rate,
          flags.tch_learning_rate_decay_factor,
          flags.tch_num_epochs_per_decay,
          tch_scope)

      pre_losses = self.get_pre_losses()
      self.pre_loss = tf.add_n(pre_losses, '%s_pre_loss' % tch_scope)
      pre_optimizer = utils.get_opt(flags, self.learning_rate, opt_epsilon=flags.tch_opt_epsilon)
      ## no clipping
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=self.global_step)

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