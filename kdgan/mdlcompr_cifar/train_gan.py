from kdgan import config
from kdgan import utils
from flags import flags
from dis_model import DIS
from std_model import STD
from data_utils import CIFAR

import tensorflow as tf
import math
import time

cifar = CIFAR(flags)

tn_dis = DIS(flags, is_training=True)
tn_std = STD(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, is_training=False)
vd_gen = STD(flags, is_training=False)

init_op = tf.global_variables_initializer()

# tot_params = 0
# for var in tf.trainable_variables():
#   num_params = 1
#   for dim in var.shape:
#     num_params *= dim.value
#   print('%-64s (%d params)' % (var.name, num_params))
#   tot_params += num_params
# print('%-64s (%d params)' % ('gan', tot_params))

def main(_):
  bst_acc = 0.0
  start_time = time.time()
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_dis.saver.restore(sess, flags.dis_model_ckpt)
    tn_std.saver.restore(sess, flags.std_model_ckpt)

    ini_dis = cifar.compute_acc(sess, vd_dis)
    ini_gen = cifar.compute_acc(sess, vd_gen)

    print('ini dis=%.4f ini gen=%.4f' % (ini_dis, ini_gen))

    batch_d, batch_g = -1, -1
    for epoch in range(flags.num_epoch):
      num_batch_d = math.ceil(flags.num_dis_epoch * flags.train_size / flags.batch_size)
      for _ in range(num_batch_d):
        batch_d += 1
        tn_image_d, label_dat_d = cifar.next_batch(sess)
        feed_dict = {tn_std.image_ph:tn_image_d}
        label_std_d = sess.run(tn_std.labels, feed_dict=feed_dict)
        sample_std_d, std_label_d = utils.gan_dis_sample(flags, label_dat_d, label_std_d)
        feed_dict = {
          tn_dis.image_ph:tn_image_d,
          tn_dis.std_sample_ph:sample_std_d,
          tn_dis.std_label_ph:std_label_d,
        }
        sess.run(tn_std.gan_train, feed_dict=feed_dict)

        if (batch_d + 1) % eval_interval != 0:
          continue
        print('dis #batch=%d' % (batch_d))

      exit()
      for dis_epoch in range(flags.num_dis_epoch):
        num_batch_d = math.ceil(tn_size / flags.batch_size)
        for image_np_d, label_dat_d in dis_datagen.generate(batch_size=flags.batch_size):
          batch_d += 1
          feed_dict = {tn_std.image_ph:image_np_d}
          label_std_d, = sess.run([tn_std.labels], feed_dict=feed_dict)
          # print('label_dat_d={} label_std_d={}'.format(label_dat_d.shape, label_std_d.shape))
          sample_np_d, label_np_d = utils.gan_dis_sample_dev(flags, label_dat_d, label_std_d)

          _, summary_d = sess.run([tn_dis.gan_update, dis_summary_op], feed_dict=feed_dict)

      for gen_epoch in range(flags.num_gen_epoch):
        # print('epoch %03d gen_epoch %03d' % (epoch, gen_epoch))
        num_batch_g = math.ceil(tn_size / flags.batch_size)
        for image_np_g, label_dat_g in gen_datagen.generate(batch_size=flags.batch_size):
        # for _ in range(num_batch_g):
        #   image_np_g, label_dat_g = gen_mnist.train.next_batch(flags.batch_size)
          batch_g += 1
          feed_dict = {tn_std.image_ph:image_np_g}
          label_gen_g, = sess.run([tn_std.labels], feed_dict=feed_dict)
          sample_np_g = utils.generate_label(flags, label_dat_g, label_gen_g)
          # sample_np_g, rescale_np_g = utils.generate_label(flags, label_dat_g, label_gen_g)
          # print(sample_np_g.shape, rescale_np_g.shape)
          feed_dict = {
            tn_dis.image_ph:image_np_g,
            tn_dis.sample_ph:sample_np_g,
          }
          reward_np_g, = sess.run([tn_dis.rewards], feed_dict=feed_dict)
          # reward_np_g *= rescale_np_g
          # print(reward_np_g)
          feed_dict = {
            tn_std.image_ph:image_np_g,
            tn_std.sample_ph:sample_np_g,
            tn_std.reward_ph:reward_np_g,
          }
          _, summary_g = sess.run([tn_std.gan_update, gen_summary_op], feed_dict=feed_dict)


          if flags.collect_cr_data:
            feed_dict = {
              vd_gen.image_ph:gen_mnist.test.images,
              vd_gen.hard_label_ph:gen_mnist.test.labels,
            }
            acc = sess.run(vd_gen.accuracy, feed_dict=feed_dict)
            acc_list.append(acc)
            if (batch_g + 1) % eval_interval != 0:
              continue
          else:
            if (batch_g + 1) % eval_interval != 0:
              continue
            feed_dict = {
              vd_gen.image_ph:gen_mnist.test.images,
              vd_gen.hard_label_ph:gen_mnist.test.labels,
            }
            acc = sess.run(vd_gen.accuracy, feed_dict=feed_dict)

          bst_acc = max(acc, bst_acc)
          tot_time = time.time() - start
          global_step = sess.run(tn_std.global_step)
          avg_time = (tot_time / global_step) * (tn_size / flags.batch_size)
          print('#%08d curacc=%.4f curbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
              (batch_g, acc, bst_acc, tot_time, avg_time))

          if acc <= bst_acc:
            continue
          # save gen parameters if necessary
  tot_time = time.time() - start_time
  print('#cifar=%d final=%.4f et=%.0fs' % (flags.train_size, bst_acc, tot_time))

if __name__ == '__main__':
    tf.app.run()









