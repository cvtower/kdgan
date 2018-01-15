from kdgan import config
from kdgan import utils

import tensorflow as tf
from nets import nets_factory
from tensorflow.contrib import slim

class TCH():
  def __init__(self, flags, dataset, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32,
        shape=(None, flags.image_size, flags.image_size, flags.channels))
    self.hard_label_ph = tf.placeholder(tf.int32, shape=(None,))

    tch_scope = 'tch'
    with tf.variable_scope(tch_scope) as scope:
      network_fn = nets_factory.get_network_fn(flags.model_name,
          num_classes=dataset.num_classes,
          weight_decay=flags.weight_decay,
          is_training=is_training)
      assert flags.image_size==network_fn.default_image_size
      self.logits, end_points = network_fn(self.image_ph)

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
    self.learning_rate = utils.get_lr(flags, self.global_step, dataset.num_samples, tch_scope)

    encoded_labels = slim.one_hot_encoding(self.hard_label_ph, dataset.num_classes)
    tf.losses.sigmoid_cross_entropy(encoded_labels, self.logits)
    pre_losses = tf.get_collection(tf.GraphKeys.LOSSES)
    print('#loss=%d' % (len(pre_losses)))
    pre_losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    print('#loss=%d' % (len(pre_losses)))
    self.pre_loss = tf.add_n(pre_losses, '%s_pre_loss' % tch_scope)
    pre_optimizer = utils.get_opt(flags, self.learning_rate)
    self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=self.global_step)
