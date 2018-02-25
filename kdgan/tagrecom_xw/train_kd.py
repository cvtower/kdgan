from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from gen_model import GEN
from tch_model import TCH
import data_utils

import math
import os
import time
import numpy as np
import tensorflow as tf
from os import path
from tensorflow.contrib import slim

tn_size = utils.get_tn_size(flags.dataset)
vd_size = utils.get_vd_size(flags.dataset)
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
vd_num_batch = int(vd_size / config.valid_batch_size)
print('#tn_size=%d #vd_size=%d' % (tn_size, vd_size))
eval_interval = int(tn_size / flags.batch_size)

tn_gen = GEN(flags, is_training=True)
tn_tch = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, is_training=False)
vd_tch = TCH(flags, is_training=False)

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))

tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate)
tf.summary.scalar(tn_gen.kd_loss.name, tn_gen.kd_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

yfccdata = data_utils.YFCCDATA(flags)
yfcceval = data_utils.YFCCEVAL(flags)

def main(_):
  best_prec, bst_epk = 0.0, 0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_gen.saver.restore(sess, flags.gen_model_ckpt)
    tn_tch.saver.restore(sess, flags.tch_model_ckpt)
    init_prec = yfcceval.compute_prec(flags, sess, vd_gen)
    print('init@%d=%.4f' % (flags.cutoff, init_prec))

    start = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_text_np, hard_label_np = yfccdata.next_batch(flags, sess)

      feed_dict = {tn_tch.image_ph:tn_image_np, tn_tch.text_ph:tn_text_np}
      soft_logit_np = sess.run(tn_tch.logits, feed_dict=feed_dict)

      feed_dict = {
        tn_gen.image_ph:tn_image_np,
        tn_gen.hard_label_ph:hard_label_np,
        tn_gen.soft_logit_ph:soft_logit_np,
      }
      _, summary = sess.run([tn_gen.kd_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      if (tn_batch + 1) % eval_interval != 0:
          continue
      prec = yfcceval.compute_prec(flags, sess, vd_gen)
      if prec > best_prec:
        bst_epk = int((tn_batch * batch_size) / tn_size)
      best_prec = max(prec, best_prec)
      tot_time = time.time() - start
      global_step = sess.run(tn_gen.global_step)
      avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
      print('#%08d prec@%d=%.4f best@%d=%.4f tot=%.0fs avg=%.2fs/epoch' % 
          (global_step, flags.cutoff, prec, bst_epk, best_prec, tot_time, avg_time))
      if prec < best_prec:
        continue
      # save if necessary
  tot_time = time.time() - start
  print('best@%d=%.4f et=%.0fs' % (flags.cutoff, best_prec, tot_time))

if __name__ == '__main__':
    tf.app.run()


