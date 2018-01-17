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
tf.app.flags.DEFINE_float('kd_lamda', 0.3, '')
tf.app.flags.DEFINE_float('temperature', 3.0, '')
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
    one_hot=True,
    validation_size=0,
    reshape=False)
print('tn size=%d vd size=%d' % (mnist.train.num_examples, mnist.test.num_examples))
tn_num_batch = int(flags.num_epoch * mnist.train.num_examples / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))
eval_interval = int(mnist.train.num_examples / flags.batch_size)
print('ev #interval=%d' % (eval_interval))

tn_gen = GEN(flags, mnist.train, is_training=True)
tn_tch = TCH(flags, mnist.train, is_training=True)

kd_scope = 'kd'
with tf.variable_scope(kd_scope):
  gen_logits = tf.scalar_mul(1.0 / flags.temperature, tn_gen.logits)
  tch_logits = tf.scalar_mul(1.0 / flags.temperature, tn_tch.logits)
  hard_loss = tf.losses.softmax_cross_entropy(tn_gen.hard_label_ph, gen_logits)
  soft_loss = tf.losses.mean_squared_error(tch_logits, gen_logits)
  kd_loss = (flags.kd_lamda * hard_loss + (1 - flags.kd_lamda) * soft_loss) / flags.batch_size
  kd_loss = tf.identity(kd_loss, name='kd_loss')

  global_step = tf.Variable(0, trainable=False)
  learning_rate = utils.get_lr(flags, global_step, mnist.train.num_examples, kd_scope)
  kd_optimizer = utils.get_opt(flags, learning_rate)
  kd_update = kd_optimizer.minimize(kd_loss, global_step=global_step)

tf.summary.scalar(learning_rate.name, learning_rate)
tf.summary.scalar(kd_loss.name, kd_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, mnist.test, is_training=False)
vd_tch = TCH(flags, mnist.test, is_training=False)

def main(_):
  gen_model_ckpt = utils.get_latest_ckpt(flags.gen_checkpoint_dir)
  tch_model_ckpt = utils.get_latest_ckpt(flags.tch_checkpoint_dir)

  for variable in tf.trainable_variables():
    num_params = 1
    for dim in variable.shape:
      num_params *= dim.value
    print('%-50s (%d params)' % (variable.name, num_params))

  best_hit_v = -np.inf
  start = time.time()
  with tf.train.MonitoredTrainingSession() as sess:
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    sess.run(init_op)
    tn_gen.saver.restore(sess, gen_model_ckpt)
    tn_tch.saver.restore(sess, tch_model_ckpt)
    gen_acc = metric.eval_mdlcompr(sess, vd_gen, mnist)
    tch_acc = metric.eval_mdlcompr(sess, vd_tch, mnist)
    tot_time = time.time() - start
    print('init gen_acc=%.4f tch_acc=%.4f time=%.0fs' % (gen_acc, tch_acc, tot_time))
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = mnist.train.next_batch(flags.batch_size)
      feed_dict = {
        tn_gen.image_ph:tn_image_np,
        tn_gen.hard_label_ph:tn_label_np,
        tn_tch.image_ph:tn_image_np,
      }
      _, summary = sess.run([kd_loss, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      if (tn_batch + 1) % eval_interval != 0:
        continue
      gen_acc = metric.eval_mdlcompr(sess, vd_gen, mnist)
      tot_time = time.time() - start
      print('#%08d gen_acc=%.4f time=%.0fs' % (tn_batch, gen_acc, tot_time))

      if acc_v < best_acc_v:
        continue
      best_acc_v = acc_v
      global_step, = sess.run([tn_gen.global_step])
      print('#%08d gen_acc=%.4f time=%.0fs' % (global_step, best_acc_v, tot_time))
      # tn_gen.saver.save(utils.get_session(sess), flags.save_path, global_step=global_step)
  print('bstacc=%.4f' % (best_acc_v))

if __name__ == '__main__':
    tf.app.run()









