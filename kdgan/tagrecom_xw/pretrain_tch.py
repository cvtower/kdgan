from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from tch_model import TCH

import os
import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

tn_size = utils.get_tn_size(flags.dataset)
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
vd_size = utils.get_vd_size(flags.dataset)
vd_num_batch = int(vd_size / config.valid_batch_size)
eval_interval = int(tn_size / flags.batch_size)
print('#batch=%d #interval=%d' % (tn_num_batch, eval_interval))

tn_tch = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_tch = TCH(flags, is_training=False)

tf.summary.scalar(tn_tch.learning_rate.name, tn_tch.learning_rate)
tf.summary.scalar(tn_tch.pre_loss.name, tn_tch.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))

tn_data_sources = utils.get_data_sources(flags, is_training=True)
vd_data_sources = utils.get_data_sources(flags, is_training=False)
print('tn: #tfrecord=%d\nvd: #tfrecord=%d' % (len(tn_data_sources), len(vd_data_sources)))

tn_ts_list = utils.decode_tfrecord(flags, tn_data_sources, shuffle=True)
vd_ts_list = utils.decode_tfrecord(flags, vd_data_sources, shuffle=False)
tn_bt_list = utils.generate_batch(tn_ts_list, flags.batch_size)
vd_bt_list = utils.generate_batch(vd_ts_list, config.valid_batch_size)
_, tn_image_bt, tn_text_bt, tn_label_bt, _ = tn_bt_list
_, vd_image_bt, vd_text_bt, vd_label_bt, _ = vd_bt_list

def main(_):
  bst_hit = 0.0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    start = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_text_np, tn_label_np = sess.run([tn_image_bt, tn_text_bt, tn_label_bt])
      feed_dict = {
        tn_tch.text_ph:tn_text_np,
        tn_tch.image_ph:tn_image_np,
        tn_tch.hard_label_ph:tn_label_np,
      }
      _, summary = sess.run([tn_tch.pre_update, summary_op], 
          feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      if (tn_batch + 1) % eval_interval != 0:
          continue
      hit = []
      for vd_batch in range(vd_num_batch):
        vd_text_np, vd_image_np, vd_label_np = sess.run([vd_text_bt, vd_image_bt, vd_label_bt])
        feed_dict = {
          vd_tch.text_ph:vd_text_np, 
          vd_tch.image_ph:vd_image_np,
        }
        vd_logit_np = sess.run(vd_tch.logits, feed_dict=feed_dict)
        hit_bt = metric.compute_hit(vd_logit_np, vd_label_np, flags.cutoff)
        hit.append(hit_bt)
      hit = np.mean(hit)
      bst_hit = max(hit, bst_hit)
      tot_time = time.time() - start
      global_step = sess.run(tn_gen.global_step)
      avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
      print('#%08d curhit=%.4f curbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
          (tn_batch, hit, bst_hit, tot_time, avg_time))

      if hit < bst_hit:
        continue
      tn_tch.saver.save(sess, flags.tch_model_ckpt)
  print('bsthit=%.4f' % (bst_hit))

if __name__ == '__main__':
  tf.app.run()