from kdgan import config
from kdgan import metric
from kdgan import utils
from dis_model import DIS
from gen_model import GEN

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
tf.app.flags.DEFINE_float('dis_keep_prob', 0.88, '')
tf.app.flags.DEFINE_float('gen_keep_prob', 0.88, '')
tf.app.flags.DEFINE_float('tch_keep_prob', 0.50, '')
tf.app.flags.DEFINE_float('kd_hard_pct', 0.7, '')
tf.app.flags.DEFINE_float('temperature', 3.0, '')
tf.app.flags.DEFINE_string('dis_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('gen_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('tch_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('tch_model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
# optimization
tf.app.flags.DEFINE_float('dis_weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('gen_weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('clip_norm', 10.0, '')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.0, '')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, '')
# tf.app.flags.DEFINE_float('opt_epsilon', 1e-8, '')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'adam|rmsprop|sgd')
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('min_learning_rate', 0.0001, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'fixed|polynomial')
flags = tf.app.flags.FLAGS

mnist = input_data.read_data_sets(flags.dataset_dir,
    one_hot=True,
    validation_size=0,
    reshape=True)
print('tn size=%d vd size=%d' % (mnist.train.num_examples, mnist.test.num_examples))
tn_num_batch = int(flags.num_epoch * mnist.train.num_examples / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))
eval_interval = int(mnist.train.num_examples / flags.batch_size)
print('ev #interval=%d' % (eval_interval))

tn_dis = DIS(flags, mnist.train, is_training=True)
tn_gen = GEN(flags, mnist.train, is_training=True)
tf.summary.scalar(tn_dis.learning_rate.name, tn_dis.learning_rate)
tf.summary.scalar(tn_dis.gan_loss.name, tn_dis.gan_loss)
tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate)
tf.summary.scalar(tn_gen.gan_loss.name, tn_gen.gan_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, mnist.test, is_training=False)
vd_gen = GEN(flags, mnist.test, is_training=False)

def main(_):
  # dis_model_ckpt = utils.get_latest_ckpt(flags.dis_checkpoint_dir)
  gen_model_ckpt = utils.get_latest_ckpt(flags.gen_checkpoint_dir)

  # for variable in tf.trainable_variables():
  #   num_params = 1
  #   for dim in variable.shape:
  #     num_params *= dim.value
  #   print('%-50s (%d params)' % (variable.name, num_params))

  bst_gen_acc = -np.inf
  start = time.time()
  with tf.train.MonitoredTrainingSession() as sess:
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    sess.run(init_op)
    tn_dis.saver.restore(sess, dis_model_ckpt)
    tn_gen.saver.restore(sess, gen_model_ckpt)
    # dis_acc = metric.eval_mdlcompr(sess, vd_dis, mnist)
    gen_acc = metric.eval_mdlcompr(sess, vd_gen, mnist)
    tot_time = time.time() - start
    # print('init dis_acc=%.4f gen_acc=%.4f time=%.0fs' % (dis_acc, gen_acc, tot_time))
    print('init gen_acc=%.4f time=%.0fs' % (gen_acc, tot_time))
  print('bstacc=%.4f' % (bst_gen_acc))

if __name__ == '__main__':
    tf.app.run()









