from kdgan import config, utils

from nets import nets_factory
from nets import vgg

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class TCH():
  def __init__(self, flags, is_training=True):
    self.is_training = is_training
    
    # None = batch_size
    self.image_ph = tf.placeholder(tf.float32, shape=(None, flags.feature_size), name="tch_image")
    self.text_ph = tf.placeholder(tf.int64, shape=(None, None), name="tch_text")
    self.hard_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label), name="tch_hard_label")
    self.soft_label_ph = tf.placeholder(tf.float32, shape=(None, flags.num_label), name="tch_soft_label")

    # None = batch_size * sample_size
    self.sample_ph = tf.placeholder(tf.int32, shape=(None, 2), name="tch_sample")
    self.reward_ph = tf.placeholder(tf.float32, shape=(None,), name="tch_reward")

    tch_scope = 'tch'
    vocab_size = utils.get_vocab_size(flags.dataset)
    model_scope = nets_factory.arg_scopes_map[flags.model_name]
    with tf.variable_scope(tch_scope) as scope:
      
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
        self.labels = tf.nn.softmax(self.logits)

    if not is_training:
      return

    save_dict = {}
    for variable in tf.trainable_variables():
      if not variable.name.startswith(tch_scope):
        continue
      print('%-50s added to TCH saver' % variable.name)
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
        tch_scope)

    # pre train
    pre_losses = []
    pre_losses.append(tf.losses.sigmoid_cross_entropy(self.hard_label_ph, self.logits))
    pre_losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.pre_loss = tf.add_n(pre_losses, name='%s_pre_loss' % tch_scope)
    pre_optimizer = tf.train.AdamOptimizer(self.learning_rate)
    #pre_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.pre_update = pre_optimizer.minimize(self.pre_loss, global_step=global_step)

    """
    # kd train
    kd_losses = self.get_kd_losses(flags)
    self.kd_loss = tf.add_n(kd_losses, name='%s_kd_loss' % gen_scope)
    kd_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.kd_update = kd_optimizer.minimize(self.kd_loss, global_step=global_step)
    """

    # kdgan train
    sample_logits = tf.gather_nd(self.logits, self.sample_ph)
    gan_losses = [tf.losses.sigmoid_cross_entropy(self.reward_ph, sample_logits)]
    kd_losses = self.get_kd_losses(flags)
    #kd_losses = 0
    #kdgan_losses = gan_losses + kd_losses
    kdgan_losses = kd_losses
    self.kdgan_loss = tf.add_n(kdgan_losses, name='%s_kdgan_loss' % tch_scope)
    kdgan_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    self.kdgan_update = kdgan_optimizer.minimize(self.kdgan_loss, global_step=global_step)

  def get_kd_losses(self, flags):
    hard_loss = flags.kd_lamda * tf.losses.sigmoid_cross_entropy(
        self.hard_label_ph, self.logits)
    
    smooth_labels = tf.nn.softmax(self.soft_label_ph / flags.temperature)
    soft_loss = (1.0 - flags.kd_lamda) * tf.nn.l2_loss(
        tf.nn.softmax(self.logits) - smooth_labels)

    kd_losses = [hard_loss, soft_loss]
    return kd_losses

