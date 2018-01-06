from kdgan import config, metric, utils
from tch_model import TCH

import os
import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_float('init_learning_rate', 0.1, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 10.0, '')
tf.app.flags.DEFINE_float('weight_decay', 0.001, 'l2 coefficient')

tf.app.flags.DEFINE_integer('cutoff', 3, '')
tf.app.flags.DEFINE_integer('embedding_size', 10, '')
tf.app.flags.DEFINE_integer('feature_size', 4096, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')

tf.app.flags.DEFINE_string('model_name', None, '')

flags = tf.app.flags.FLAGS

num_batch_t = int(flags.num_epoch * config.train_data_size / config.train_batch_size)
num_batch_v = int(config.valid_data_size / config.valid_batch_size)
print('tn: #batch={}\nvd: #batch={}'.format(num_batch_t, num_batch_v))

def main(_):
  global_step = tf.train.create_global_step()
  gen_t = TCH(flags, is_training=True)
  scope = tf.get_variable_scope()
  scope.reuse_variables()
  gen_v = TCH(flags, is_training=False)

  for variable in tf.trainable_variables():
    num_params = 1
    for dim in variable.shape:
      num_params *= dim.value
    print('{}\t({} params)'.format(variable.name, num_params))

if __name__ == '__main__':
  tf.app.run()