from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from data_utils import AffineGenerator
from dis_model import DIS
from gen_model import GEN
from tch_model import TCH
import data_utils

from os import path
from tensorflow.contrib import slim
import math
import os
import time
import numpy as np
import tensorflow as tf

dis_mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
dis_datagen = AffineGenerator(dis_mnist)
gen_mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
gen_datagen = AffineGenerator(gen_mnist)
tch_mnist = data_utils.read_data_sets(flags.dataset_dir,
    one_hot=True,
    train_size=flags.train_size,
    valid_size=flags.valid_size,
    reshape=True)
tch_datagen = AffineGenerator(tch_mnist)

tn_size, vd_size = gen_mnist.train.num_examples, gen_mnist.test.num_examples
print('tn size=%d vd size=%d' % (tn_size, vd_size))
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
print('tn #batch=%d' % (tn_num_batch))
eval_interval = int(tn_size / flags.batch_size)
print('ev #interval=%d' % (eval_interval))

tn_dis = DIS(flags, dis_mnist.train, is_training=True)
tn_gen = GEN(flags, gen_mnist.train, is_training=True)
tn_tch = TCH(flags, tch_mnist.train, is_training=True)
dis_summary_op = tf.summary.merge([
  tf.summary.scalar(tn_dis.learning_rate.name, tn_dis.learning_rate),
  tf.summary.scalar(tn_dis.gan_loss.name, tn_dis.gan_loss),
])
gen_summary_op = tf.summary.merge([
  tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate),
  tf.summary.scalar(tn_gen.gan_loss.name, tn_gen.gan_loss),
])
init_op = tf.global_variables_initializer()

scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, dis_mnist.test, is_training=False)
vd_gen = GEN(flags, gen_mnist.test, is_training=False)
vd_tch = TCH(flags, tch_mnist.test, is_training=False)

# for variable in tf.trainable_variables():
#   num_params = 1
#   for dim in variable.shape:
#     num_params *= dim.value
#   print('%-50s (%d params)' % (variable.name, num_params))

def main(_):
  bst_gen_acc, bst_tch_acc = 0.0, 0.0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_dis.saver.restore(sess, flags.dis_model_ckpt)
    tn_gen.saver.restore(sess, flags.gen_model_ckpt)
    tn_tch.saver.restore(sess, flags.tch_model_ckpt)

    feed_dict = {
      vd_dis.image_ph:dis_mnist.test.images,
      vd_dis.hard_label_ph:dis_mnist.test.labels,
    }
    ini_dis = sess.run(vd_dis.accuracy, feed_dict=feed_dict)
    feed_dict = {
      vd_gen.image_ph:gen_mnist.test.images,
      vd_gen.hard_label_ph:gen_mnist.test.labels,
    }
    ini_gen = sess.run(vd_gen.accuracy, feed_dict=feed_dict)
    print('ini dis=%.4f ini gen=%.4f' % (ini_dis, ini_gen))
    # exit()

    start = time.time()
    batch_d, batch_g, batch_t = -1, -1, -1
    for epoch in range(flags.num_epoch):
      for dis_epoch in range(flags.num_dis_epoch):
        # print('epoch %03d dis_epoch %03d' % (epoch, dis_epoch))
        # num_batch_d = math.ceil(tn_size / flags.batch_size)
        # for _ in range(num_batch_d):
        #   image_d, label_dat_d = dis_mnist.train.next_batch(flags.batch_size)
        for image_d, label_dat_d in dis_datagen.generate(batch_size=flags.batch_size):
          batch_d += 1

          # feed_dict = {tn_gen.image_ph:image_d}
          # label_gen_d = sess.run(tn_gen.labels, feed_dict=feed_dict)
          # sample_gen_d, dis_label_gen = utils.gan_dis_sample(flags, label_dat_d, label_gen_d)
          # feed_dict = {
          #   tn_dis.image_ph:image_d,
          #   tn_dis.sample_ph:sample_gen_d,
          #   tn_dis.dis_label_ph:dis_label_gen,
          # }
          # sess.run(tn_dis.gan_update, feed_dict=feed_dict)

          feed_dict = {tn_tch.image_ph:image_d}
          label_tch_d = sess.run(tn_tch.labels, feed_dict=feed_dict)
          sample_tch_d, dis_label_tch = utils.gan_dis_sample(flags, label_dat_d, label_tch_d)
          feed_dict = {
            tn_dis.image_ph:image_d,
            tn_dis.sample_ph:sample_tch_d,
            tn_dis.dis_label_ph:dis_label_tch,
          }
          sess.run(tn_dis.gan_update, feed_dict=feed_dict)

      for tch_epoch in range(flags.num_tch_epoch):
        # num_batch_t = math.ceil(tn_size / flags.batch_size)
        # for _ in range(num_batch_t):
        #   image_t, label_dat_t = tch_mnist.train.next_batch(flags.batch_size)
        for image_t, label_dat_t in tch_datagen.generate(batch_size=flags.batch_size):
          batch_t += 1

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

          if flags.kdgan_model != config.kdgan_ow_flag:
            feed_dict = {vd_gen.image_ph:image_t}
            soft_logit_t = sess.run(vd_gen.logits, feed_dict=feed_dict)
            feed_dict = {
              tn_tch.image_ph:image_t,
              tn_tch.sample_ph:sample_t,
              tn_tch.reward_ph:reward_t,
              tn_tch.hard_label_ph:label_dat_t,
              tn_tch.soft_logit_ph:soft_logit_t,
            }

          sess.run(tn_tch.kdgan_update, feed_dict=feed_dict)

          if (batch_t + 1) % eval_interval != 0:
            continue
          feed_dict = {
            vd_tch.image_ph:gen_mnist.test.images,
            vd_tch.hard_label_ph:gen_mnist.test.labels,
          }
          tch_acc = sess.run(vd_tch.accuracy, feed_dict=feed_dict)

          bst_tch_acc = max(tch_acc, bst_tch_acc)
          print('#%08d tchcur=%.4f tchbst=%.4f' % (batch_t, tch_acc, bst_tch_acc))

      for gen_epoch in range(flags.num_gen_epoch):
        # print('epoch %03d gen_epoch %03d' % (epoch, gen_epoch))
        # num_batch_g = math.ceil(tn_size / flags.batch_size)
        # for _ in range(num_batch_g):
        #   image_g, label_dat_g = gen_mnist.train.next_batch(flags.batch_size)
        for image_g, label_dat_g in gen_datagen.generate(batch_size=flags.batch_size):
          batch_g += 1

          feed_dict = {tn_gen.image_ph:image_g}
          label_gen_g = sess.run(tn_gen.labels, feed_dict=feed_dict)
          sample_g = utils.generate_label(flags, label_dat_g, label_gen_g)
          feed_dict = {
            tn_dis.image_ph:image_g,
            tn_dis.sample_ph:sample_g,
          }
          reward_g = sess.run(tn_dis.rewards, feed_dict=feed_dict)

          feed_dict = {vd_tch.image_ph:image_g}
          soft_logit_g = sess.run(vd_tch.logits, feed_dict=feed_dict)

          feed_dict = {
            tn_gen.image_ph:image_g,
            tn_gen.sample_ph:sample_g,
            tn_gen.reward_ph:reward_g,
            tn_gen.hard_label_ph:label_dat_g,
            tn_gen.soft_logit_ph:soft_logit_g,
          }
          sess.run(tn_gen.kdgan_update, feed_dict=feed_dict)
          
          if (batch_g + 1) % eval_interval != 0:
            continue
          feed_dict = {
            vd_gen.image_ph:gen_mnist.test.images,
            vd_gen.hard_label_ph:gen_mnist.test.labels,
          }
          gen_acc = sess.run(vd_gen.accuracy, feed_dict=feed_dict)

          bst_gen_acc = max(gen_acc, bst_gen_acc)
          tot_time = time.time() - start
          global_step = sess.run(tn_gen.global_step)
          avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
          print('#%08d gencur=%.4f genbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
              (batch_g, gen_acc, bst_gen_acc, tot_time, avg_time))

          if gen_acc <= bst_gen_acc:
            continue
          # save gen parameters if necessary
  tot_time = time.time() - start
  print('#mnist=%d kdgan_%s=%.4f et=%.0fs' % 
      (tn_size, flags.kdgan_model, bst_gen_acc, tot_time))

if __name__ == '__main__':
    tf.app.run()









