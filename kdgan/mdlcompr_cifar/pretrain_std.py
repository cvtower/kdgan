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
    start = time.time()
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = cifar.train.next_batch(flags.batch_size)
      # print(tn_image_np.shape, tn_label_np.shape)

if __name__ == '__main__':
  tf.app.run()
