from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from dis_model import DIS

import os
import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

tn_size = utils.get_tn_size(flags.dataset)
vd_size = utils.get_vd_size(flags.dataset)
print('#tn_data=%d #vd_data=%d' % (tn_size, vd_size))
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
vd_num_batch = int(vd_size / config.valid_batch_size)
print('#tn_batch=%d #vd_batch=%d' % (tn_num_batch, vd_num_batch))
eval_interval = int(tn_size / flags.batch_size)

dis_t = DIS(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
dis_v = DIS(flags, is_training=False)

tf.summary.scalar(dis_t.learning_rate.name, dis_t.learning_rate)
tf.summary.scalar(dis_t.pre_loss.name, dis_t.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))

data_sources_t = utils.get_data_sources(flags, is_training=True)
print('#tn_tfrecord=%d' % (len(data_sources_t)))

ts_list_t = utils.decode_tfrecord(flags, data_sources_t, shuffle=True)
bt_list_t = utils.generate_batch(ts_list_t, flags.batch_size)
user_bt_t, image_bt_t, text_bt_t, label_bt_t, file_bt_t = bt_list_t

vd_image_np, vd_text_np, vd_label_np, _ = utils.get_valid_data(flags)

def main(_):
  start = time.time()
  best_hit_v = -np.inf
  with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    with slim.queues.QueueRunners(sess):
      for batch_t in range(tn_num_batch):
        text_np_t, image_np_t, label_np_t = sess.run([text_bt_t, image_bt_t, label_bt_t])
        feed_dict = {dis_t.text_ph:text_np_t, dis_t.image_ph:image_np_t, dis_t.hard_label_ph:label_np_t}
        _, summary = sess.run([dis_t.pre_update, summary_op], feed_dict=feed_dict)
        writer.add_summary(summary, batch_t)

        if (batch_t + 1) % eval_interval != 0:
            continue
        feed_dict = {
          dis_v.image_ph:vd_image_np,
          dis_v.text_ph:vd_text_np,
        }
        vd_logit_np = sess.run(dis_v.logits, feed_dict=feed_dict)
        vd_hit = metric.compute_hit(vd_logit_np, vd_label_np, flags.cutoff)

        tot_time = time.time() - start
        print('#%08d hit=%.4f %06ds' % (batch_t, vd_hit, int(tot_time)))

        if vd_hit < best_hit_v:
          continue
        best_hit_v = vd_hit
        dis_t.saver.save(sess, flags.dis_model_ckpt)
  print('best hit=%.4f' % (best_hit_v))

if __name__ == '__main__':
  tf.app.run()