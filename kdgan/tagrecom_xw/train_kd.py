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
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
eval_interval = int(tn_size / flags.batch_size)
print('tn: #data=%d #batch=%d' % (tn_size, tn_num_batch))

tn_gen = GEN(flags, is_training=True)
tn_tch = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, is_training=False)
vd_tch = TCH(flags, is_training=False)

tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate)
tf.summary.scalar(tn_gen.kd_loss.name, tn_gen.kd_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))

yfcc100m = data_utils.YFCC100M(flags)
vd_image_np, vd_text_np, vd_label_np, _ = utils.get_valid_data(flags)

def main(_):
  bst_hit = 0.0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_gen.saver.restore(sess, flags.gen_model_ckpt)
    tn_tch.saver.restore(sess, flags.tch_model_ckpt)

    feed_dict = {vd_gen.image_ph:vd_image_np}
    vd_logit_np = sess.run(vd_gen.logits, feed_dict=feed_dict)
    ini_hit = metric.compute_hit(vd_logit_np, vd_label_np, flags.cutoff)
    print('inihit=%.4f' % (ini_hit))

    feed_dict = {
      vd_tch.image_ph:vd_image_np,
      vd_tch.text_ph:vd_text_np,
    }
    vd_logit_np = sess.run(vd_tch.logits, feed_dict=feed_dict)
    tch_hit = metric.compute_hit(vd_logit_np, vd_label_np, flags.cutoff)
    print('tchhit=%.4f' % (tch_hit))

    exit()

    start = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_text_np, tn_hard_label_np = yfcc100m.next_batch(flags, sess)
      # print('hard labels:\t{}'.format(tn_hard_label_np.shape))
      # print(np.argsort(-tn_hard_label_np[0,:])[:10])

      feed_dict = {
        tn_tch.image_ph:tn_image_np,
        tn_tch.text_ph:tn_text_np,
      }
      tn_soft_logit_np = sess.run(tn_tch.logits, feed_dict=feed_dict)
      # print('soft labels:\t{}'.format(tn_soft_logit_np.shape))
      # print(np.argsort(-tn_soft_logit_np[0,:])[:10])

      feed_dict = {
        tn_gen.image_ph:tn_image_np,
        tn_gen.hard_label_ph:tn_hard_label_np,
        tn_gen.soft_logit_ph:tn_soft_logit_np,
      }
      _, summary = sess.run([tn_gen.kd_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      if (tn_batch + 1) % eval_interval != 0:
          continue
      feed_dict = {vd_gen.image_ph:vd_image_np}
      vd_logit_np = sess.run(vd_gen.logits, feed_dict=feed_dict)
      hit = metric.compute_hit(vd_logit_np, vd_label_np, flags.cutoff)
      bst_hit = max(hit, bst_hit)
      tot_time = time.time() - start
      global_step = sess.run(tn_gen.global_step)
      avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
      print('#%08d curhit=%.4f curbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
          (tn_batch, hit, bst_hit, tot_time, avg_time))

      if hit < bst_hit:
        continue
      # save if necessary
  tot_time = time.time() - start
  print('bsthit=%.4f et=%.0fs' % (bst_hit, tot_time))

if __name__ == '__main__':
    tf.app.run()


