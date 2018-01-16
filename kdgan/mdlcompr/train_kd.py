from kdgan import config
from kdgan import metric
from kdgan import utils
from gen_model import GEN
from tch_model import TCH

from os import path
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
import math
import os
import time
import numpy as np
import tensorflow as tf

# dataset
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_integer('channels', 1, '')
tf.app.flags.DEFINE_integer('image_size', 28, '')
tf.app.flags.DEFINE_integer('num_label', 10, '')
# model
tf.app.flags.DEFINE_float('gen_keep_prob', 0.95, '')
tf.app.flags.DEFINE_float('tch_keep_prob', 0.5, '')
tf.app.flags.DEFINE_string('gen_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('tch_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('tch_model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
# optimization
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('clip_norm', 10.0, '')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.0, '')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, '')
tf.app.flags.DEFINE_float('opt_epsilon', 1e-6, '')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_string('optimizer', 'rmsprop', 'adam|sgd')
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('min_learning_rate', 0.0001, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'fixed|polynomial')
flags = tf.app.flags.FLAGS

mnist = input_data.read_data_sets(flags.dataset_dir,
    one_hot=False,
    validation_size=0,
    reshape=False)
print('tn size=%d vd size=%d' % (mnist.train.num_examples, mnist.test.num_examples))
tn_num_batch = int(flags.num_epoch * mnist.train.num_examples / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))
eval_interval = int(mnist.train.num_examples / flags.batch_size)
print('ev #interval=%d' % (eval_interval))

tn_gen = GEN(flags, mnist.train, is_training=True)
tn_tch = TCH(flags, mnist.train, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, mnist.test, is_training=False)
vd_tch = TCH(flags, mnist.test, is_training=False)

def main(_):
  # print('gen_checkpoint_dir=%s' % (flags.gen_checkpoint_dir))
  # print('tch_checkpoint_dir=%s' % (flags.tch_checkpoint_dir))
  gen_ckpt = utils.get_latest_ckpt(flags.gen_checkpoint_dir)
  # print('gen_ckpt=%s' % (gen_ckpt))
  tch_ckpt = utils.get_latest_ckpt(flags.tch_checkpoint_dir)
  # print('tch_ckpt=%s' % (tch_ckpt))

  for variable in tf.trainable_variables():
    num_params = 1
    for dim in variable.shape:
      num_params *= dim.value
    print('%-50s (%d params)' % (variable.name, num_params))

if __name__ == '__main__':
    tf.app.run()









