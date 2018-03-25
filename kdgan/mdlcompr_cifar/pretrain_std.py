from kdgan import config
from flags import flags
from std_model import STD
import cifar10_utils

from datetime import datetime
import numpy as np
import tensorflow as tf
import math
import time

cifar10_utils.maybe_download_and_extract()
with tf.device('/cpu:0'):
  tn_image_ts, tn_label_ts = cifar10_utils.distorted_inputs()
  vd_image_ts, vd_label_ts = cifar10_utils.inputs(eval_data=True)
tn_num_batch = int(flags.num_epoch * flags.train_size / flags.batch_size)
vd_num_batch = int(math.ceil(flags.valid_size / flags.batch_size))
print('#tn_batch=%d #vd_batch=%d' % (tn_num_batch, vd_num_batch))
eval_interval = int(math.ceil(flags.train_size / flags.batch_size))

tn_std = STD(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_std = STD(flags, is_training=False)
init_op = tf.global_variables_initializer()

def main(argv=None):
  bst_acc = 0.0
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    start_time = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = sess.run([tn_image_ts, tn_label_ts])
      feed_dict = {tn_std.image_ph:tn_image_np, tn_std.hard_label_ph:tn_label_np}
      sess.run(tn_std.pre_train, feed_dict=feed_dict)
      if (tn_batch + 1) % eval_interval != 0 and (tn_batch + 1) != tn_num_batch:
        continue
      acc_list = []
      for vd_batch in range(vd_num_batch):
        vd_image_np, vd_label_np = sess.run([vd_image_ts, vd_label_ts])
        feed_dict = {vd_std.image_ph:vd_image_np, vd_std.hard_label_ph:vd_label_np}
        acc = sess.run(vd_std.accuracy, feed_dict=feed_dict)
        acc_list.append(acc)
      acc = sum(acc_list) / len(acc_list)
      bst_acc = max(acc, bst_acc)

      end_time = time.time()
      duration = end_time - start_time
      avg_time = duration / (tn_batch + 1)
      print('#batch=%d acc=%.4f time=%.4fs/batch est=%.4fh' % 
          (tn_batch + 1, bst_acc, avg_time, avg_time * tn_num_batch / 3600))
  print('final=%.4f' % (bst_acc))

if __name__ == '__main__':
  tf.app.run()










