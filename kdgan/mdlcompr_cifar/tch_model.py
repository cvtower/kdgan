from resnet_utils import HParams, ResNet

import tensorflow as tf

class TCH():
  def __init__(self, flags, is_training=True):
    hps = HParams(batch_size=flags.batch_size,
        num_classes=flags.num_label,
        lrn_rate=flags.tch_learning_rate,
        min_lrn_rate=flags.min_learning_rate,
        num_residual_units=3,
        use_bottleneck=False,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
        optimizer='mom')

    self.image_ph = image_ph = tf.placeholder(tf.float32, 
        shape=(flags.batch_size, flags.image_size, flags.image_size, flags.channels))
    self.hard_label_ph = hard_label_ph = tf.placeholder(tf.float32, 
        shape=(flags.batch_size, flags.num_label))

    self.tch_scope = tch_scope = 'tch'
    with tf.variable_scope(tch_scope) as scope:
      if not is_training:
        model = ResNet(hps, image_ph, hard_label_ph, 'valid', tch_scope)
      else:
        model = ResNet(hps, image_ph, hard_label_ph, 'train', tch_scope)
      model.build_graph()
      self.predictions = model.predictions
      if not is_training:
        return
      self.saver = model.saver
      self.global_step = model.global_step
      self.lrn_rate = model.lrn_rate
      self.pre_train = model.train_op


