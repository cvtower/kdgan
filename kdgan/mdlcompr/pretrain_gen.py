from kdgan import config
from kdgan import metric
from kdgan import utils
from gen_model import GEN
import data_utils

from os import path
from tensorflow.contrib import slim
import time
import numpy as np
import tensorflow as tf

# dataset
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_integer('image_size', 28, '')
tf.app.flags.DEFINE_integer('channels', 1, '')
# model
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_string('checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('save_path', None, '')
tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
# optimization
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, '')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, '')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_string('optimizer', 'rmsprop', 'adam|sgd')
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('min_learning_rate', 0.0001, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'fixed|polynomial')
flags = tf.app.flags.FLAGS

tn_dataset = data_utils.get_dataset(flags, is_training=True)
vd_dataset = data_utils.get_dataset(flags, is_training=False)
tn_image_bt, tn_label_bt = data_utils.generate_batch(flags, tn_dataset, is_training=True)
vd_image_bt, vd_label_bt = data_utils.generate_batch(flags, vd_dataset, is_training=False)

tn_gen = GEN(flags, tn_dataset, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, vd_dataset, is_training=False)

def main(_):
  pass

if __name__ == '__main__':
  tf.app.run()
