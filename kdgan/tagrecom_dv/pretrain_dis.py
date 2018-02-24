from kdgan import config
from kdgan import metric
from kdgan import utils
from dis_model import DIS

import os
import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('dataset', None, '')
tf.app.flags.DEFINE_integer('num_label', 100, '')
# evaluation
tf.app.flags.DEFINE_integer('cutoff', 3, '')
# image model
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_integer('feature_size', 4096, '')
tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('image_model', None, '')
# training
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('num_epoch', 20, '')
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95, '')
tf.app.flags.DEFINE_float('end_learning_rate', 0.00001, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 10.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exp', 'fix|ply')
# dis model
tf.app.flags.DEFINE_float('dis_weight_decay', 0.0, 'l2 coefficient')
tf.app.flags.DEFINE_string('dis_model_ckpt', None, '')
tf.app.flags.DEFINE_integer('num_dis_epoch', 10, '')
# gen model
tf.app.flags.DEFINE_float('kd_lamda', 0.3, '')
tf.app.flags.DEFINE_float('gen_weight_decay', 0.001, 'l2 coefficient')
tf.app.flags.DEFINE_float('temperature', 3.0, '')
tf.app.flags.DEFINE_string('gen_model_ckpt', None, '')
tf.app.flags.DEFINE_integer('num_gen_epoch', 5, '')
# tch model
tf.app.flags.DEFINE_float('tch_weight_decay', 0.00001, 'l2 coefficient')
tf.app.flags.DEFINE_integer('embedding_size', 10, '')
tf.app.flags.DEFINE_string('tch_model_ckpt', None, '')
tf.app.flags.DEFINE_integer('num_tch_epoch', 5, '')
flags = tf.app.flags.FLAGS

train_data_size = utils.get_tn_size(flags.dataset)
valid_data_size = utils.get_vd_size(flags.dataset)
num_batch_t = int(flags.num_epoch * train_data_size / flags.batch_size)
num_batch_v = int(valid_data_size / config.valid_batch_size)
eval_interval = int(train_data_size / flags.batch_size)
print('tn:\t#batch=%d\nvd:\t#batch=%d\neval:\t#interval=%d' % (
    num_batch_t, num_batch_v, eval_interval))

def main(_):
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
  data_sources_v = utils.get_data_sources(flags, is_training=False)
  print('tn: #tfrecord=%d\nvd: #tfrecord=%d' % (len(data_sources_t), len(data_sources_v)))
  
  ts_list_t = utils.decode_tfrecord(flags, data_sources_t, shuffle=True)
  ts_list_v = utils.decode_tfrecord(flags, data_sources_v, shuffle=False)
  bt_list_t = utils.generate_batch(ts_list_t, flags.batch_size)
  bt_list_v = utils.generate_batch(ts_list_v, config.valid_batch_size)
  user_bt_t, image_bt_t, text_bt_t, label_bt_t, file_bt_t = bt_list_t
  user_bt_v, image_bt_v, text_bt_v, label_bt_v, file_bt_v = bt_list_v

  start = time.time()
  best_hit_v = -np.inf
  with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    with slim.queues.QueueRunners(sess):
      for batch_t in range(num_batch_t):
        #image_np_t, label_np_t = sess.run([image_bt_t, label_bt_t])
        text_np_t, image_np_t, label_np_t = sess.run([text_bt_v, image_bt_v, label_bt_v])
        feed_dict = {dis_t.text_ph:text_np_t, dis_t.image_ph:image_np_t, dis_t.hard_label_ph:label_np_t}
        _, summary = sess.run([dis_t.pre_update, summary_op], feed_dict=feed_dict)
        writer.add_summary(summary, batch_t)

        if (batch_t + 1) % eval_interval != 0:
            continue

        hit_v = []
        for batch_v in range(num_batch_v):
          text_np_v, image_np_v, label_np_v = sess.run([text_bt_v, image_bt_v, label_bt_v])
          feed_dict = {dis_v.text_ph:text_np_v, dis_v.image_ph:image_np_v}
          logit_np_v, = sess.run([dis_v.logits], feed_dict=feed_dict)
          hit_bt = metric.compute_prec(logit_np_v, label_np_v, flags.cutoff)
          hit_v.append(hit_bt)
        hit_v = np.mean(hit_v)

        tot_time = time.time() - start
        print('#%08d hit=%.4f %06ds' % (batch_t, hit_v, int(tot_time)))

        if hit_v < best_hit_v:
          continue
        best_hit_v = hit_v
        dis_t.saver.save(sess, flags.dis_model_ckpt)
  print('best hit=%.4f' % (best_hit_v))

if __name__ == '__main__':
  tf.app.run()