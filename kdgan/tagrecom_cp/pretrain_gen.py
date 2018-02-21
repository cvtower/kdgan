from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from gen_model import GEN

from os import path
from tensorflow.contrib import slim
import os
import time
import numpy as np
import tensorflow as tf

tn_size = utils.get_tn_size(flags.dataset)
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
eval_interval = int(tn_size / flags.batch_size)
print('#batch=%d #interval=%d' % (tn_num_batch, eval_interval))
# exit()

data_sources = utils.get_data_sources(flags, is_training=True)
print('tn: #tfrecord=%d' % (len(data_sources)))
ts_list = utils.decode_tfrecord(flags, data_sources, shuffle=True)
bt_list = utils.generate_batch(ts_list, flags.batch_size)
user_bt, image_bt, text_bt, label_bt, file_bt = bt_list
vd_image_np, vd_label_np, vd_imgid_np = utils.get_valid_data(flags)

tn_gen = GEN(flags, is_training=True)
tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate)
tf.summary.scalar(tn_gen.pre_loss.name, tn_gen.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, is_training=False)

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))
# exit()

def main(_):
  bst_hit = 0.0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    start = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = sess.run([image_bt, label_bt])
      feed_dict = {tn_gen.image_ph:tn_image_np, tn_gen.hard_label_ph:tn_label_np}
      _, summary = sess.run([tn_gen.pre_update, summary_op], feed_dict=feed_dict)
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
      tn_gen.saver.save(utils.get_session(sess), flags.gen_model_ckpt)
  tot_time = time.time() - start
  print('bsthit=%.4f et=%.0fs' % (bst_hit, tot_time))

if __name__ == '__main__':
  tf.app.run()