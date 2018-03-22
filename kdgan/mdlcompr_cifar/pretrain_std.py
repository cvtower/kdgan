from kdgan import config
from flags import flags
from std_model import STD
import data_utils

import time
import numpy as np
import tensorflow as tf

cifar = data_utils.CIFAR()
tn_data_size = cifar.train.num_examples
tn_num_batch = int(flags.num_epoch * tn_data_size / flags.batch_size)
print('train #data=%d #batch=%d' % (tn_data_size, tn_num_batch))
eval_interval = int(max(tn_data_size / flags.batch_size, 1.0))

tn_std = STD(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_std = STD(flags, is_training=False)

tot_params = 0
for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))
  tot_params += num_params
print('%-50s (%d params)' % ('mlp', tot_params))

tf.summary.scalar(tn_std.learning_rate.name, tn_std.learning_rate)
tf.summary.scalar(tn_std.pre_loss.name, tn_std.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

def main(_):
  bst_acc = 0.0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    start = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = cifar.train.next_batch(flags.batch_size)
      # print(tn_image_np.shape, tn_label_np.shape)
      feed_dict = {tn_std.image_ph:tn_image_np, tn_std.hard_label_ph:tn_label_np}
      _, summary = sess.run([tn_std.pre_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)
      
      if (tn_batch + 1) % eval_interval != 0:
        continue
      feed_dict = {
        vd_std.image_ph:cifar.test.images,
        vd_std.hard_label_ph:cifar.test.labels,
      }
      acc = sess.run(vd_std.accuracy, feed_dict=feed_dict)

      bst_acc = max(acc, bst_acc)
      tot_time = time.time() - start
      global_step = sess.run(tn_std.global_step)
      avg_time = (tot_time / global_step) * (tn_data_size / flags.batch_size)
      print('#%08d curacc=%.4f curbst=%.4f tot=%.0fs avg=%.2fs/epoch' % 
          (tn_batch, acc, bst_acc, tot_time, avg_time))

      if acc < bst_acc:
        continue
      tn_std.saver.save(utils.get_session(sess), flags.std_model_ckpt)
  tot_time = time.time() - start
  print('#cifar=%d bstacc=%.4f et=%.0fs' % (tn_data_size, bst_acc, tot_time))

if __name__ == '__main__':
  tf.app.run()
