from kdgan import config
from kdgan import metric
from kdgan import utils
from gen_model import GEN
import data_utils

from os import path
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# dataset
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_integer('channels', 1, '')
tf.app.flags.DEFINE_integer('image_size', 28, '')
tf.app.flags.DEFINE_integer('num_label', 10, '')
tf.app.flags.DEFINE_string('augmentation_type', 'affine', 'none|affine|align')
# model
tf.app.flags.DEFINE_float('gen_keep_prob', 0.88, '')
tf.app.flags.DEFINE_string('checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('save_path', None, '')
# optimization
tf.app.flags.DEFINE_float('gen_weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('gen_opt_epsilon', 1e-6, '')
tf.app.flags.DEFINE_float('clip_norm', 10.0, '')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.0, '')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, '')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'adam|rmsprop|sgd')
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 1e-3, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('end_learning_rate', 1e-5, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'fixed', 'exponential|polynomial')
flags = tf.app.flags.FLAGS

SEED = 777
tf.set_random_seed(SEED)
np.random.seed(SEED)

mnist = input_data.read_data_sets(flags.dataset_dir, one_hot=True, validation_size=0)
print('tn size=%d vd size=%d' % (mnist.train.num_examples, mnist.test.num_examples))

tn_gen = GEN(flags, mnist.train, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, mnist.test, is_training=False)

tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate)
tf.summary.scalar(tn_gen.pre_loss.name, tn_gen.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

def main(_):
  utils.delete_if_exist(flags.checkpoint_dir)
  gen_ckpt = tf.train.latest_checkpoint(flags.checkpoint_dir)
  # print('gen ckpt=%s' % (gen_ckpt))
  if gen_ckpt != None:
    print('todo init from gen ckpt')
    exit()
  utils.create_if_nonexist(flags.checkpoint_dir)

  datagen = data_utils.Generator(flags.augmentation_type, mnist)

  best_acc_v, best_epoch = -np.inf, -1
  no_impr_patience = init_patience = 10
  start = time.time()
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    for tn_epoch in range(flags.num_epoch):
      for tn_image_np, tn_label_np in datagen.generate(batch_size=flags.batch_size):
        feed_dict = {tn_gen.image_ph:tn_image_np, tn_gen.hard_label_ph:tn_label_np}
        _, summary = sess.run([tn_gen.pre_update, summary_op], feed_dict=feed_dict)
        global_step, = sess.run([tn_gen.global_step])
        writer.add_summary(summary, global_step)

      vd_image_np, vd_label_np = mnist.test.images, mnist.test.labels
      feed_dict = {vd_gen.image_ph:vd_image_np}
      predictions, = sess.run([vd_gen.predictions], feed_dict=feed_dict)
      acc_v = metric.compute_acc(predictions, vd_label_np)
      tot_time = time.time() - start
      print('#%08d curacc=%.4f %.0fs' % (global_step, acc_v, tot_time))

      if acc_v <= best_acc_v:
        no_impr_patience -= 1
        if no_impr_patience == 0:
          no_impr_patience = init_patience
          _ = sess.run([tn_gen.lr_update])
        continue
      best_acc_v = acc_v
      best_epoch = tn_epoch
      print('#%08d CURBST=%.4f %.0fs' % (global_step, best_acc_v, tot_time))
      tn_gen.saver.save(utils.get_session(sess), flags.save_path, global_step=global_step)
  print('bstacc=%.4f epoch=%03d' % (best_acc_v, best_epoch))

if __name__ == '__main__':
  tf.app.run()
