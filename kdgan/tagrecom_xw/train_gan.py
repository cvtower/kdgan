from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from dis_model import DIS
from gen_model import GEN

import math
import os
import time
import numpy as np
import tensorflow as tf
from os import path
from tensorflow.contrib import slim

tn_size = utils.get_tn_size(flags.dataset)
eval_interval = int(tn_size / flags.batch_size)
print('#tn_data=%d' % (tn_size))

tn_dis = DIS(flags, is_training=True)
tn_gen = GEN(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, is_training=False)
vd_gen = GEN(flags, is_training=False)

dis_summary_op = tf.summary.merge([
  tf.summary.scalar(tn_dis.learning_rate.name, tn_dis.learning_rate),
  tf.summary.scalar(tn_dis.gan_loss.name, tn_dis.gan_loss),
])
gen_summary_op = tf.summary.merge([
  tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate),
  tf.summary.scalar(tn_gen.gan_loss.name, tn_gen.gan_loss),
])
init_op = tf.global_variables_initializer()

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))

yfccdata = data_utils.YFCCDATA(flags)
yfcceval = data_utils.YFCCEVAL(flags)

def main(_):
  best_prec = -np.inf
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_dis.saver.restore(sess, flags.dis_model_ckpt)
    tn_gen.saver.restore(sess, flags.gen_model_ckpt)
    init_prec = yfcceval.compute_prec(flags, sess, vd_gen)
    print('init prec=%.4f' % (init_prec))
    exit()
    start = time.time()
    batch_d, batch_g = -1, -1
    for epoch in range(flags.num_epoch):
      for dis_epoch in range(flags.num_dis_epoch):
        print('epoch %03d dis_epoch %03d' % (epoch, dis_epoch))
        num_batch_d = math.ceil(train_data_size / flags.batch_size)
        for _ in range(num_batch_d):
          batch_d += 1
          image_np_d, label_dat_d = sess.run([image_bt_d, label_bt_d])
          feed_dict = {tn_gen.image_ph:image_np_d}
          label_gen_d, = sess.run([tn_gen.labels], feed_dict=feed_dict)
          sample_np_d, label_np_d = utils.gan_dis_sample(
              flags, label_dat_d, label_gen_d)
          feed_dict = {
            tn_dis.image_ph:image_np_d,
            tn_dis.sample_ph:sample_np_d,
            tn_dis.dis_label_ph:label_np_d,
          }
          _, summary_d = sess.run([tn_dis.gan_update, dis_summary_op], 
              feed_dict=feed_dict)
          writer.add_summary(summary_d, batch_d)

      for gen_epoch in range(flags.num_gen_epoch):
        print('epoch %03d gen_epoch %03d' % (epoch, gen_epoch))
        num_batch_g = math.ceil(train_data_size / flags.batch_size)
        for _ in range(num_batch_g):
          batch_g += 1
          image_np_g, label_dat_g = sess.run([image_bt_g, label_bt_g])
          feed_dict = {tn_gen.image_ph:image_np_g}
          label_gen_g, = sess.run([tn_gen.labels], feed_dict=feed_dict)
          sample_np_g = utils.generate_label(
              flags, label_dat_g, label_gen_g)
          feed_dict = {
            tn_dis.image_ph:image_np_g,
            tn_dis.sample_ph:sample_np_g,
          }
          reward_np_g, = sess.run([tn_dis.rewards], feed_dict=feed_dict)
          feed_dict = {
            tn_gen.image_ph:image_np_g,
            tn_gen.sample_ph:sample_np_g,
            tn_gen.reward_ph:reward_np_g,
          }
          _, summary_g = sess.run([tn_gen.gan_update, gen_summary_op], 
              feed_dict=feed_dict)
          writer.add_summary(summary_g, batch_g)
          
          # if (batch_g + 1) % eval_interval != 0:
          #   continue
          # prec = utils.evaluate(flags, sess, vd_gen, bt_list_v)
          # tot_time = time.time() - start
          # print('#%08d hit=%.4f %06ds' % (batch_g, prec, int(tot_time)))
          # if prec <= best_prec:
          #   continue
          # best_prec = prec
          # print('best hit=%.4f' % (best_prec))
      prec = utils.evaluate(flags, sess, vd_gen, bt_list_v)
      tot_time = time.time() - start
      print('#%03d curbst=%.4f %.0fs' % (epoch, prec, tot_time))
      figure_data.append((epoch, prec))
      if prec <= best_prec:
        continue
      best_prec = prec
  tot_time = time.time() - start
  print('best@%d=%.4f et=%.0fs' % (flags.cutoff, best_prec, tot_time))

if __name__ == '__main__':
  tf.app.run()






