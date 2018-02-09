from kdgan import config
from kdgan import metric
from kdgan import utils
from dis_model import DIS
import data_utils

from os import path
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np
import tensorflow as tf

# dataset
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_integer('channels', 1, '')
tf.app.flags.DEFINE_integer('image_size', 28, '')
tf.app.flags.DEFINE_integer('num_label', 10, '')
# model
tf.app.flags.DEFINE_float('dis_keep_prob', 0.88, '')
tf.app.flags.DEFINE_string('dis_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('dis_save_path', None, '')
# optimization
tf.app.flags.DEFINE_float('dis_weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('dis_opt_epsilon', 1e-6, '')
tf.app.flags.DEFINE_float('clip_norm', 10.0, '')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.0, '')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, '')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_string('optimizer', 'rmsprop', 'adam|sgd')
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('end_learning_rate', 0.0001, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'fixed|polynomial')
flags = tf.app.flags.FLAGS

mnist = input_data.read_data_sets(flags.dataset_dir,
    one_hot=True,
    validation_size=0,
    reshape=True)
print('tn size=%d vd size=%d' % (mnist.train.num_examples, mnist.test.num_examples))
tn_num_batch = int(flags.num_epoch * mnist.train.num_examples / flags.batch_size)
vd_num_batch = int(mnist.train.num_examples / config.valid_batch_size)
print('tn #batch=%d vd #batch=%d' % (tn_num_batch, vd_num_batch))
eval_interval = int(mnist.train.num_examples / flags.batch_size)
print('ev #interval=%d' % (eval_interval))

tn_dis = DIS(flags, mnist.train, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_dis = DIS(flags, mnist.test, is_training=False)

tf.summary.scalar(tn_dis.learning_rate.name, tn_dis.learning_rate)
tf.summary.scalar(tn_dis.pre_loss.name, tn_dis.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

def main(_):
  utils.delete_if_exist(flags.dis_checkpoint_dir)
  dis_ckpt = tf.train.latest_checkpoint(flags.dis_checkpoint_dir)
  # print('dis ckpt=%s' % (dis_ckpt))
  if dis_ckpt != None:
    print('todo init from dis ckpt')
    exit()
  utils.create_if_nonexist(flags.dis_checkpoint_dir)

  best_acc_v = -np.inf
  start = time.time()
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = mnist.train.next_batch(flags.batch_size)
      feed_dict = {tn_dis.image_ph:tn_image_np, tn_dis.hard_label_ph:tn_label_np}
      _, summary = sess.run([tn_dis.pre_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      if (tn_batch + 1) % eval_interval != 0:
        continue
      vd_image_np, vd_label_np = mnist.test.images, mnist.test.labels
      feed_dict = {vd_dis.image_ph:vd_image_np}
      predictions, = sess.run([vd_dis.predictions], feed_dict=feed_dict)
      acc_v = metric.compute_acc(predictions, vd_label_np)
      tot_time = time.time() - start
      # print('#%08d hit=%.4f %06ds' % (tn_batch, acc_v, int(tot_time)))

      if acc_v < best_acc_v:
        continue
      best_acc_v = acc_v
      global_step, = sess.run([tn_dis.global_step])
      print('#%08d acc=%.4f %.0fs' % (global_step, best_acc_v, tot_time))
      tn_dis.saver.save(utils.get_session(sess), flags.dis_save_path, global_step=global_step)
  print('bstacc=%.4f' % (best_acc_v))

if __name__ == '__main__':
  tf.app.run()
