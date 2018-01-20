from kdgan import config
from kdgan import utils

import tensorflow as tf
from nets import nets_factory
from tensorflow.contrib import slim

class DIS():
  def __init__(self, flags, dataset, is_training=True):
    self.is_training = is_training
    
    num_feature = flags.image_size * flags.image_size * flags.channels
    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32, shape=(None, num_feature))
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label))

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.dis_label_ph = tf.placeholder(tf.float32, shape=(None,))

    hidden_size = 800
    self.dis_scope = dis_scope = 'dis'
    with tf.variable_scope(dis_scope):
      self.logits = utils.build_mlp_logits(flags, self.image_ph,
        hidden_size=hidden_size,
        keep_prob=flags.dis_keep_prob,
        weight_decay=flags.dis_weight_decay,
        is_training=is_training)
      
      sample_logits = tf.gather_nd(self.logits, self.sample_ph)
      reward_logits = self.logits
      # reward_logits = 2 * (tf.sigmoid(reward_logits) - 0.5)
      reward_logits = tf.sigmoid(reward_logits)
      self.rewards = tf.gather_nd(reward_logits, self.sample_ph)

      if not is_training:
        self.predictions = tf.argmax(self.logits, axis=1)
        return

      save_dict, var_list = {}, []
      for variable in tf.trainable_variables():
        if not variable.name.startswith(dis_scope):
          continue
        print('%-50s added to DIS saver' % variable.name)
        save_dict[variable.name] = variable
        var_list.append(variable)
      self.saver = tf.train.Saver(save_dict)

      self.global_step = tf.Variable(0, trainable=False)
      self.learning_rate = utils.get_lr(flags, self.global_step, dataset.num_examples, dis_scope)

      # pre train
      pre_losses = self.get_pre_losses()
      self.pre_loss = tf.add_n(pre_losses, '%s_pre_loss' % dis_scope)
      pre_optimizer = utils.get_opt(flags, self.learning_rate, opt_epsilon=flags.dis_opt_epsilon)
      self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=self.global_step)

      # gan train
      gan_losses = []
      gan_losses.append(tf.losses.sigmoid_cross_entropy(self.dis_label_ph, sample_logits))
      # gan_losses.extend(self.get_regularization_losses())
      self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % dis_scope)
      gan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
      self.gan_update = gan_optimizer.minimize(self.gan_loss, global_step=self.global_step)

  def get_regularization_losses(self):
    regularization_losses = []
    for regularization_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
      if not regularization_loss.name.startswith(self.dis_scope):
        continue
      regularization_losses.append(regularization_loss)
    return regularization_losses

  def get_pre_losses(self):
    pre_losses = [tf.losses.softmax_cross_entropy(self.hard_label_ph, self.logits)]
    print('#pre_losses=%d' % (len(pre_losses)))
    pre_losses.extend(self.get_regularization_losses())
    print('#pre_losses=%d' % (len(pre_losses)))
    return pre_losses
















