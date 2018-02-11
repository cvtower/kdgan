from kdgan import config
from kdgan import metric
from kdgan import utils
from dis_model import DIS
from gen_model import GEN
from tch_model import TCH
from flags import flags
import data_utils

from os import path
from tensorflow.contrib import slim
import math
import os
import time
import numpy as np
import tensorflow as tf

mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)

tn_size, vd_size = mnist.train.num_examples, mnist.test.num_examples
print('tn size=%d vd size=%d' % (tn_size, vd_size))
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))
eval_interval = int(tn_size / flags.batch_size)
print('ev #interval=%d' % (eval_interval))

tn_dis = DIS(flags, mnist.train, is_training=True)
tn_gen = GEN(flags, mnist.train, is_training=True)
tn_tch = TCH(flags, mnist.train, is_training=True)
init_op = tf.global_variables_initializer()

scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, mnist.test, is_training=False)
vd_gen = GEN(flags, mnist.test, is_training=False)
vd_tch = TCH(flags, mnist.test, is_training=False)

# for variable in tf.trainable_variables():
#   num_params = 1
#   for dim in variable.shape:
#     num_params *= dim.value
#   print('%-50s (%d params)' % (variable.name, num_params))

def main(_):
  bst_acc = 0.0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_dis.saver.restore(sess, flags.dis_ckpt_file)
    tn_gen.saver.restore(sess, flags.gen_ckpt_file)
    tn_tch.saver.restore(sess, flags.tch_ckpt_file)

    feed_dict = {
      vd_dis.image_ph:mnist.test.images,
      vd_dis.hard_label_ph:mnist.test.labels,
    }
    ini_dis = sess.run(vd_dis.accuracy, feed_dict=feed_dict)
    feed_dict = {
      vd_gen.image_ph:mnist.test.images,
      vd_gen.hard_label_ph:mnist.test.labels,
    }
    ini_gen = sess.run(vd_gen.accuracy, feed_dict=feed_dict)
    feed_dict = {
      vd_tch.image_ph:mnist.test.images,
      vd_tch.hard_label_ph:mnist.test.labels,
    }
    ini_tch = sess.run(vd_tch.accuracy, feed_dict=feed_dict)
    print('ini dis=%.4f gen=%.4f tch=%.4f' % (ini_dis, ini_gen, ini_tch))
    # exit()

    start = time.time()
    batch_d, batch_g, batch_t = -1, -1, -1
    for epoch in range(flags.num_epoch):
      for dis_epoch in range(flags.num_dis_epoch):
        # print('epoch %03d dis_epoch %03d' % (epoch, dis_epoch))
        num_batch_d = math.ceil(tn_size / flags.batch_size)
        for _ in range(num_batch_d):
          batch_d += 1
          image_d, label_dat_d = mnist.train.next_batch(flags.batch_size)

          feed_dict = {tn_gen.image_ph:image_d}
          label_gen_d = sess.run(tn_gen.labels, feed_dict=feed_dict)
          feed_dict = {tn_tch.image_ph:image_d}
          label_tch_d = sess.run(tn_tch.labels, feed_dict=feed_dict)

          sample_d, label_d = utils.kdgan_dis_sample(flags, label_dat_d, label_gen_d, label_tch_d)
          feed_dict = {
            tn_dis.image_ph:image_d,
            tn_dis.sample_ph:sample_d,
            tn_dis.dis_label_ph:label_d,
          }
          sess.run(tn_dis.gan_update, feed_dict=feed_dict)

      for tch_epoch in range(flags.num_tch_epoch):
        num_batch_t = math.ceil(tn_size / flags.batch_size)
        for _ in range(num_batch_t):
          batch_t += 1
          image_t, label_dat_t = mnist.train.next_batch(flags.batch_size)

          feed_dict = {tn_tch.image_ph:image_t}
          label_tch_t = sess.run(tn_tch.labels, feed_dict=feed_dict)
          sample_t = utils.generate_label(flags, label_dat_t, label_tch_t)
          feed_dict = {
            tn_dis.image_ph:image_t,
            tn_dis.sample_ph:sample_t,
          }
          reward_t = sess.run(tn_dis.rewards, feed_dict=feed_dict)
          feed_dict = {
            tn_tch.image_ph:image_t,
            tn_tch.sample_ph:sample_t,
            tn_tch.reward_ph:reward_t,
          }
          sess.run(tn_tch.kdgan_update, feed_dict=feed_dict)

      for gen_epoch in range(flags.num_gen_epoch):
        # print('epoch %03d gen_epoch %03d' % (epoch, gen_epoch))
        num_batch_g = math.ceil(tn_size / flags.batch_size)
        for _ in range(num_batch_g):
          batch_g += 1
          image_g, label_dat_g = mnist.train.next_batch(flags.batch_size)

          feed_dict = {tn_tch.image_ph:image_g}
          soft_logits = sess.run(tn_tch.logits, feed_dict=feed_dict)

          feed_dict = {tn_gen.image_ph:image_g}
          label_gen_g = sess.run(tn_gen.labels, feed_dict=feed_dict)
          sample_g = utils.generate_label(flags, label_dat_g, label_gen_g)
          # sample_g, rescale_np_g = utils.generate_label(flags, label_dat_g, label_gen_g)
          # print(sample_g.shape, rescale_np_g.shape)
          feed_dict = {
            tn_dis.image_ph:image_g,
            tn_dis.sample_ph:sample_g,
          }
          reward_g = sess.run(tn_dis.rewards, feed_dict=feed_dict)
          # reward_g *= rescale_np_g
          # print(reward_g)
          
          feed_dict = {
            tn_gen.image_ph:image_g,
            tn_gen.hard_label_ph:label_dat_g,
            tn_gen.soft_logit_ph:soft_logits,
            tn_gen.sample_ph:sample_g,
            tn_gen.reward_ph:reward_g,
          }
          sess.run(tn_gen.kdgan_update, feed_dict=feed_dict)

          if (batch_g + 1) % eval_interval != 0:
            continue
          feed_dict = {
            vd_gen.image_ph:mnist.test.images,
            vd_gen.hard_label_ph:mnist.test.labels,
          }
          acc = sess.run(vd_gen.accuracy, feed_dict=feed_dict)

          bst_acc = max(acc, bst_acc)
          tot_time = time.time() - start
          global_step = sess.run(tn_gen.global_step)
          avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
          print('#%08d curacc=%.4f curbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
              (batch_g, acc, bst_acc, tot_time, avg_time))

          if acc <= bst_acc:
            continue
          # save gen parameters if necessary
  print('#mnist=%d bstacc=%.4f' % (tn_size, bst_acc))

if __name__ == '__main__':
    tf.app.run()









