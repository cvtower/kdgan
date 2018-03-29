from flags import flags
from std_model import STD
from tch_model import TCH
from data_utils import CIFAR

import tensorflow as tf
import math
import time

cifar = CIFAR(flags)

tn_num_batch = int(flags.num_epoch * flags.train_size / flags.batch_size)
print('#tn_batch=%d' % (tn_num_batch))
eval_interval = int(math.ceil(flags.train_size / flags.batch_size))


tn_std = STD(flags, is_training=True)
tn_tch = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_std = STD(flags, is_training=False)
vd_tch = TCH(flags, is_training=False)

init_op = tf.global_variables_initializer()

# tot_params = 0
# for var in tf.trainable_variables():
#   num_params = 1
#   for dim in var.shape:
#     num_params *= dim.value
#   print('%-64s (%d params)' % (var.name, num_params))
#   tot_params += num_params
# print('%-64s (%d params)' % (' '.join(['kd', flags.kd_model]), tot_params))

def main(_):
  bst_acc = 0.0
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_std.saver.restore(sess, flags.std_model_ckpt)
    ini_std = cifar.compute_acc(sess, vd_std)
    tf.logging.info('ini_std=%.4f' % (ini_std))
    tn_tch.saver.restore(sess, flags.tch_model_ckpt)
    ini_tch = cifar.compute_acc(sess, vd_tch)
    tf.logging.info('ini_tch=%.4f' % (ini_tch))

    start_time = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = cifar.next_batch(sess)
      
      feed_dict = {tn_tch.image_ph:tn_image_np}
      soft_label_np = sess.run(tn_tch.labels, feed_dict=feed_dict)
      print(soft_label_np.shape)

      exit()

  tf.logging.info('#cifar=%d final=%.4f' % (flags.train_size, bst_acc))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

