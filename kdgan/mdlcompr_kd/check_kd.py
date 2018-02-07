from kdgan import config
from kdgan import metric
from kdgan import utils
from gen_model import GEN
from tch_model import TCH
import data_utils

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
tf.app.flags.DEFINE_integer('train_size', 60000, '')
tf.app.flags.DEFINE_integer('valid_size', 0, '')
# model
tf.app.flags.DEFINE_float('gen_keep_prob', 0.95, '')
tf.app.flags.DEFINE_float('tch_keep_prob', 0.5, '')
tf.app.flags.DEFINE_float('kd_hard_pct', 0.3, '')
tf.app.flags.DEFINE_float('temperature', 3.0, '')
tf.app.flags.DEFINE_string('gen_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('tch_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('tch_model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
# optimization
tf.app.flags.DEFINE_float('gen_weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('gen_opt_epsilon', 1e-6, '')
tf.app.flags.DEFINE_float('tch_weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('tch_opt_epsilon', 1e-6, '')
tf.app.flags.DEFINE_float('clip_norm', 10.0, '')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.0, '')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, '')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'rmsprop|sgd')
# learning rate
tf.app.flags.DEFINE_float('gen_learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('gen_learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('gen_num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_float('tch_learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('tch_learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('tch_num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'fixed|polynomial')
flags = tf.app.flags.FLAGS

mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
print('tn size=%d vd size=%d' % (mnist.train.num_examples, mnist.test.num_examples))
tn_num_batch = int(flags.num_epoch * mnist.train.num_examples / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))
eval_interval = int(mnist.train.num_examples / flags.batch_size)
print('ev #interval=%d' % (eval_interval))

tn_tch = TCH(flags, mnist.train, is_training=True)
tn_gen = GEN(flags, mnist.train, is_training=True)
tn_sl_gen = GEN(flags, mnist.train, is_training=True, gen_scope='sl_gen')
tn_kd_gen = GEN(flags, mnist.train, is_training=True, gen_scope='kd_gen')
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_tch = TCH(flags, mnist.test, is_training=False)
vd_gen = GEN(flags, mnist.test, is_training=False)
vd_sl_gen = GEN(flags, mnist.test, is_training=False, gen_scope='sl_gen')
vd_kd_gen = GEN(flags, mnist.test, is_training=False, gen_scope='kd_gen')

tf.summary.scalar(tn_sl_gen.learning_rate.name, tn_sl_gen.learning_rate)
tf.summary.scalar(tn_kd_gen.learning_rate.name, tn_kd_gen.learning_rate)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

# for variable in tf.trainable_variables():
#   num_params = 1
#   for dim in variable.shape:
#     num_params *= dim.value
#   print('%-50s (%d params)' % (variable.name, num_params))

init_sl_gen_ckpt = path.join(flags.gen_checkpoint_dir, 'sl_gen')
init_kd_gen_ckpt = path.join(flags.gen_checkpoint_dir, 'kd_gen')

def ini():
  start = time.time()
  with tf.Session() as sess:
    sess.run(init_op)
    ini_acc = metric.eval_mdlcompr(sess, vd_gen, mnist)
    # print('%-50s:%.4f' % ('ini', ini_acc))

    for var, sl_var, kd_var in zip(tn_gen.var_list, tn_sl_gen.var_list, tn_kd_gen.var_list):
      # print('%-50s\n%-50s\n%-50s' % (var.name, sl_var.name, kd_var.name))
      var_value = sess.run(var)
      sl_assign = sl_var.assign(var_value)
      sess.run(sl_assign)
      kd_assign = kd_var.assign(var_value)
      sess.run(kd_assign)

    # ini_sl_acc = metric.eval_mdlcompr(sess, vd_sl_gen, mnist)
    # ini_kd_acc = metric.eval_mdlcompr(sess, vd_kd_gen, mnist)
    # print('%-50s:%.4f\n%-50s:%.4f' % ('ini sl', ini_sl_acc, 'ini kd', ini_kd_acc))

    tn_sl_gen.saver.save(sess, init_sl_gen_ckpt)
    tn_kd_gen.saver.save(sess, init_kd_gen_ckpt)
  tot_time = time.time() - start

def run():
  start = time.time()
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_sl_gen.saver.restore(sess, init_sl_gen_ckpt)
    tn_kd_gen.saver.restore(sess, init_kd_gen_ckpt)
    ini_sl_acc = metric.eval_mdlcompr(sess, vd_sl_gen, mnist)
    ini_kd_acc = metric.eval_mdlcompr(sess, vd_kd_gen, mnist)
    print('%-50s:%.4f\n%-50s:%.4f' % ('ini sl', ini_sl_acc, 'ini kd', ini_kd_acc))
  tot_time = time.time() - start

def main(_):
  ini()
  run()

if __name__ == '__main__':
    tf.app.run()









