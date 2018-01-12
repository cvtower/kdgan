from kdgan import config
from kdgan import metric
from kdgan import utils
from dis_model import DIS
from gen_model import GEN
from tch_model import TCH

import math
import os
import time
import numpy as np
import tensorflow as tf
from os import path
from tensorflow.contrib import slim

tf.app.flags.DEFINE_integer('batch_size', 32, '')
# learning rate configuration
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95, '')
tf.app.flags.DEFINE_float('min_learning_rate', 0.00001, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 10.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type',
    'exponential', 'fixed|exponential|polynomial')
flags = tf.app.flags.FLAGS

dis_t = DIS(flags, is_training=True)
# gen_t = GEN(flags, is_training=True)
# tch_t = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
dis_v = DIS(flags, is_training=False)
# gen_v = GEN(flags, is_training=False)
# tch_v = TCH(flags, is_training=False)

def main(_):
  for variable in tf.trainable_variables():
    num_params = 1
    for dim in variable.shape:
      num_params *= dim.value
    print('{}\t({} params)'.format(variable.name, num_params))

if __name__ == '__main__':
  tf.app.run()






