from kdgan import config
from kdgan import utils

from nets import nets_factory
from nets import vgg

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class DIS():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training

    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32, shape=(None, flags.feature_size), name="dis_image")
    self.text_ph = tf.placeholder(tf.int64, shape=(None, None), name="dis_text")
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label), name="dis_hard_label")
    self.soft_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label), name="dis_soft_label")

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2))
    self.dis_label_ph = tf.placeholder(tf.float32, shape=(None,))

    dis_scope = 'dis'
    model_scope = nets_factory.arg_scopes_map[flags.model_name]
    vocab_size = utils.get_vocab_size(flags.dataset)
    with tf.variable_scope(dis_scope) as scope:
      # newly added
      #with slim.arg_scope(model_scope(weight_decay=flags.gen_weight_decay)):
      with slim.arg_scope([slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(flags.tch_weight_decay)):
        #"""
        net = self.image_ph
        net = slim.dropout(net, flags.dropout_keep_prob, 
            is_training=is_training)
        #net = slim.fully_connected(net, flags.num_label,
            #activation_fn=None)
        #self.logits = net
        #"""
        #with slim.arg_scope([slim.fully_connected],
            #weights_regularizer=slim.l2_regularizer(flags.tch_weight_decay)):
        word_embedding = slim.variable('word_embedding',
            shape=[vocab_size, flags.embedding_size],
            # regularizer=slim.l2_regularizer(flags.tch_weight_decay),
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        # word_embedding = tf.get_variable('word_embedding', initializer=initializer)
        text_embedding = tf.nn.embedding_lookup(word_embedding, self.text_ph)
        text_embedding = tf.reduce_mean(text_embedding, axis=-2)
        #self.logits = slim.fully_connected(text_embedding, flags.num_label,
                  #activation_fn=None)
        self.combined_layer = tf.concat([net, text_embedding], 1)
        self.logits =slim.fully_connected(self.combined_layer, flags.num_label,
                activation_fn=None) 
        #"""
        #self.labels = tf.nn.softmax(self.logits)








      #old version
      """
      with slim.arg_scope(model_scope(weight_decay=flags.dis_weight_decay)):
        net = self.image_ph
        net = slim.dropout(net, flags.dropout_keep_prob, 
            is_training=is_training)
        net = slim.fully_connected(net, flags.num_label,
            activation_fn=None)
        self.logits = net
      """
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    reward_logits = self.logits
    # reward_logits = 2 * (tf.sigmoid(reward_logits) - 0.5)
    # reward_logits -= tf.reduce_mean(reward_logits, 1, keep_dims=True)
    # reward_logits -= tf.reduce_mean(reward_logits, 1, keep_dims=True)
    # reward_logits = 2 * (tf.sigmoid(reward_logits) - 0.5)
    reward_logits = tf.sigmoid(reward_logits)
    # reward_logits -= tf.reduce_mean(reward_logits, 1, keep_dims=True)
    self.rewards = tf.gather_nd(reward_logits, self.sample_ph)

    if not is_training:
      return

    save_dict = {}
    for variable in tf.trainable_variables():
      if not variable.name.startswith(dis_scope):
        continue
      print('%-50s added to DIS saver' % variable.name)
      save_dict[variable.name] = variable
    self.saver = tf.train.Saver(save_dict)

    global_step = tf.Variable(0, trainable=False)
    train_data_size = utils.get_tn_size(flags.dataset)
    self.learning_rate = utils.get_lr(
        flags,
        train_data_size,
        global_step,
        #train_data_size,
        flags.learning_rate,
        #flags.learning_rate_decay_factor,
        #flags.num_epochs_per_decay,
        dis_scope)
    
    # pre train
    pre_losses = []
    pre_losses.append(tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits))
    pre_losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % dis_scope)
    pre_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=global_step)

    # gan train
    gan_losses = []
    gan_losses.append(tf.losses.sigmoid_cross_entropy(self.dis_label_ph, sample_logits))
    gan_losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.gan_loss = tf.add_n(gan_losses, name='%s_gan_loss' % dis_scope)
    gan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.gan_update = gan_optimizer.minimize(self.gan_loss, global_step=global_step)


