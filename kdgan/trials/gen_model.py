from kdgan import config

from nets import nets_factory

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

class GEN():
    def __init__(self, flags, is_training=True):
        self.is_training = is_training

        self.image_ph = tf.placeholder(tf.float32,
                shape=(None, flags.feature_size))
        self.label_ph = tf.placeholder(tf.float32,
                shape=(None, config.num_label))
        
        self.logits = slim.fully_connected(self.image_ph, config.num_label,
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(flags.weight_decay),
                biases_initializer=tf.zeros_initializer(),
                scope='generator')

        if not is_training:
            return

        global_step = tf.train.create_global_step()
        decay_steps = int(config.train_data_size / config.train_batch_size * flags.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(flags.init_learning_rate,
                global_step, decay_steps, flags.learning_rate_decay_factor,
                staircase=True, name='exponential_decay_learning_rate')

        tf.losses.sigmoid_cross_entropy(self.label_ph, self.logits)
        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        losses.extend(regularization_losses)
        loss = tf.add_n(losses, name='loss')
        total_loss = tf.losses.get_total_loss(name='total_loss')
        diff = tf.subtract(loss, total_loss)


        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('diff', diff)
        self.summary_op = tf.summary.merge_all()

        variables_to_train = tf.trainable_variables()
        for variable in variables_to_train:
            num_params = 1
            for dim in variable.shape:
                num_params *= dim.value
            print('trainable {}\t({} params)'.format(variable.name, num_params))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.train_op = optimizer.minimize(loss, 
                global_step=global_step)





