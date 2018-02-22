from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from gen_model import GEN
from tch_model import TCH

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

tn_data_sources = utils.get_data_sources(flags, is_training=True, single_source=False)
# tn_data_sources = utils.get_data_sources(flags, is_training=False, single_source=False)
print('#tfrecord=%d for training' % (len(tn_data_sources)))
tn_ts_list = utils.decode_tfrecord(flags, tn_data_sources, shuffle=True)
tn_bt_list = utils.generate_batch(tn_ts_list, flags.batch_size)
tn_user_bt, tn_image_bt, tn_text_bt, tn_label_bt, _ = tn_bt_list
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
    exit()

    start = time.time()
    for batch_t in range(tn_num_batch):
      image_np_t, text_np_t, hard_labels = sess.run([tn_image_bt, tn_text_bt, tn_label_bt])
      # print('hard labels:\t{}'.format(hard_labels.shape))
      # print(np.argsort(-hard_labels[0,:])[:10])

      feed_dict = {tn_tch.text_ph:text_np_t}
      soft_labels, = sess.run([tn_tch.labels], feed_dict=feed_dict)
      # print('soft labels:\t{}'.format(soft_labels.shape))
      # print(np.argsort(-soft_labels[0,:])[:10])

      feed_dict = {
        tn_gen.image_ph:image_np_t,
        tn_gen.hard_label_ph:hard_labels,
        tn_gen.soft_label_ph:soft_labels,
      }
      _, summary = sess.run([tn_gen.kd_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, batch_t)

      if (batch_t + 1) % eval_interval != 0:
          continue
      hit_v = utils.evaluate(flags, sess, vd_gen, bt_list_v)
      tot_time = time.time() - start
      print('#%08d hit=%.4f %06ds' % (batch_t, hit_v, int(tot_time)))
      if hit_v <= bst_hit:
        continue
      bst_hit = hit_v
      print('best hit=%.4f' % (bst_hit))
  print('best hit=%.4f' % (bst_hit))

if __name__ == '__main__':
    tf.app.run()


