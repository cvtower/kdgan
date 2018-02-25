from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from dis_model import DIS
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
eval_interval = int(tn_size / flags.batch_size)
print('#tn_size=%d' % (tn_size))

dis_t = DIS(flags, is_training=True)
gen_t = GEN(flags, is_training=True)
tch_t = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
dis_v = DIS(flags, is_training=False)
gen_v = GEN(flags, is_training=False)
tch_v = TCH(flags, is_training=False)

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))

dis_summary_op = tf.summary.merge([
  tf.summary.scalar(dis_t.learning_rate.name, dis_t.learning_rate),
  tf.summary.scalar(dis_t.gan_loss.name, dis_t.gan_loss),
])
gen_summary_op = tf.summary.merge([
  tf.summary.scalar(gen_t.learning_rate.name, gen_t.learning_rate),
  tf.summary.scalar(gen_t.kdgan_loss.name, gen_t.kdgan_loss),
])
tch_summary_op = tf.summary.merge([
  tf.summary.scalar(tch_t.learning_rate.name, tch_t.learning_rate),
  tf.summary.scalar(tch_t.kdgan_loss.name, tch_t.kdgan_loss),
])
init_op = tf.global_variables_initializer()

yfccdata_d = data_utils.YFCCDATA(flags)
yfccdata_g = data_utils.YFCCDATA(flags)
yfccdata_t = data_utils.YFCCDATA(flags)
yfcceval = data_utils.YFCCEVAL(flags)

def main(_):
  best_prec = 0.0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.Session() as sess:
    sess.run(init_op)
    dis_t.saver.restore(sess, flags.dis_model_ckpt)
    gen_t.saver.restore(sess, flags.gen_model_ckpt)
    tch_t.saver.restore(sess, flags.tch_model_ckpt)
    start = time.time()
    with slim.queues.QueueRunners(sess):
      gen_hit = utils.evaluate_image(flags, sess, gen_v, bt_list_v)
      tch_hit = utils.evaluate_text(flags, sess, tch_v, bt_list_v)
      print('hit gen=%.4f tch=%.4f' % (gen_hit, tch_hit))

      batch_d, batch_g, batch_t = -1, -1, -1
      for epoch in range(flags.num_epoch):
        for dis_epoch in range(flags.num_dis_epoch):
          print('epoch %03d dis_epoch %03d' % (epoch, dis_epoch))
          for _ in range(num_batch_per_epoch):
            batch_d += 1
            image_d, text_d, label_dat_d = sess.run([image_bt_d, text_bt_d, label_bt_d])
            
            feed_dict = {gen_t.image_ph:image_d}
            label_gen_d, = sess.run([gen_t.labels], feed_dict=feed_dict)
            # print('gen label', label_gen_d.shape)
            feed_dict = {tch_t.text_ph:text_d}
            label_tch_d, = sess.run([tch_t.labels], feed_dict=feed_dict)
            # print('tch label', label_tch_d.shape)

            sample_d, label_d = utils.kdgan_dis_sample(flags, 
                label_dat_d, label_gen_d, label_tch_d)
            # print(sample_d.shape, label_d.shape)

            feed_dict = {
              dis_t.image_ph:image_d,
              dis_t.sample_ph:sample_d,
              dis_t.dis_label_ph:label_d,
            }
            _, summary_d = sess.run([dis_t.gan_update, dis_summary_op], 
                feed_dict=feed_dict)
            writer.add_summary(summary_d, batch_d)

        for tch_epoch in range(flags.num_tch_epoch):
          print('epoch %03d tch_epoch %03d' % (epoch, tch_epoch))
          for _ in range(num_batch_per_epoch):
            continue
            batch_t += 1
            image_t, text_t, label_dat_t = sess.run([image_bt_t, text_bt_t, label_bt_t])

            feed_dict = {tch_t.text_ph:text_t}
            label_tch_t, = sess.run([tch_t.labels], feed_dict=feed_dict)
            sample_t = utils.generate_label(flags, label_dat_t, label_tch_t)
            feed_dict = {
              dis_t.image_ph:image_t,
              dis_t.sample_ph:sample_t,
            }
            reward_t, = sess.run([dis_t.rewards], feed_dict=feed_dict)

            feed_dict = {
              tch_t.text_ph:text_t,
              tch_t.sample_ph:sample_t,
              tch_t.reward_ph:reward_t,
            }
            _, summary_t = sess.run([tch_t.kdgan_update, tch_summary_op], 
                feed_dict=feed_dict)
            writer.add_summary(summary_t, batch_t)

        for gen_epoch in range(flags.num_gen_epoch):
          print('epoch %03d gen_epoch %03d' % (epoch, gen_epoch))
          for _ in range(num_batch_per_epoch):
            batch_g += 1
            image_g, text_g, label_dat_g = sess.run([image_bt_g, text_bt_g, label_bt_g])

            feed_dict = {tch_t.text_ph:text_g}
            label_tch_g, = sess.run([tch_t.labels], feed_dict=feed_dict)
            # print('tch label {}'.format(label_tch_g.shape))

            feed_dict = {gen_t.image_ph:image_g}
            label_gen_g, = sess.run([gen_t.labels], feed_dict=feed_dict)
            sample_g = utils.generate_label(flags, label_dat_g, label_gen_g)
            feed_dict = {
              dis_t.image_ph:image_g,
              dis_t.sample_ph:sample_g,
            }
            reward_g, = sess.run([dis_t.rewards], feed_dict=feed_dict)

            feed_dict = {
              gen_t.image_ph:image_g,
              gen_t.hard_label_ph:label_dat_g,
              gen_t.soft_label_ph:label_tch_g,
              gen_t.sample_ph:sample_g,
              gen_t.reward_ph:reward_g,
            }
            _, summary_g = sess.run([gen_t.kdgan_update, gen_summary_op], 
                feed_dict=feed_dict)
            writer.add_summary(summary_g, batch_g)

  print('best hit=%.4f' % (best_prec))

if __name__ == '__main__':
  tf.app.run()






