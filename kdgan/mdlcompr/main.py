from kdgan import config
from kdgan import utils

import time
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
from tensorflow.contrib import slim

# optimization
tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('adam_beta1', 0.9, '')
tf.app.flags.DEFINE_float('adam_beta2', 0.999, '')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, '')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, '')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, '')
tf.app.flags.DEFINE_integer('batch_size', 32, '')
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_string('optimizer', 'rmsprop', 'adam|sgd')
# learning rate
tf.app.flags.DEFINE_float('learning_rate', 0.01, '')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, '')
tf.app.flags.DEFINE_float('min_learning_rate', 0.0001, '')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, '')
tf.app.flags.DEFINE_string('learning_rate_decay_type', 'exponential', 'fixed|polynomial')
flags = tf.app.flags.FLAGS

dataset_name = 'mnist'
tch_model_name = 'lenet'
tch_preprocessing_name = 'lenet'

num_readers = 4
num_preprocessing_threads = 4

tn_split_name = 'train'

def main(_):
  tn_dataset = dataset_factory.get_dataset(
      dataset_name,
      tn_split_name,
      flags.dataset_dir)
  tn_tch_network = nets_factory.get_network_fn(
      tch_model_name,
      num_classes=tn_dataset.num_classes,
      weight_decay=flags.weight_decay,
      is_training=True)
  tn_preprocessing = preprocessing_factory.get_preprocessing(
      tch_preprocessing_name,
      is_training=True)
  tn_provider = slim.dataset_data_provider.DatasetDataProvider(
      tn_dataset,
      num_readers=num_readers,
      common_queue_capacity=20 * flags.batch_size,
      common_queue_min=10 * flags.batch_size)
  [tn_image_ts, tn_label_ts] = tn_provider.get(['image', 'label'])
  tn_image_size = tn_tch_network.default_image_size
  tn_image_ts = tn_preprocessing(tn_image_ts, tn_image_size, tn_image_size)
  tn_image_bt, tn_label_bt = tf.train.batch(
      [tn_image_ts, tn_label_ts],
      batch_size=flags.batch_size,
      num_threads=num_preprocessing_threads,
      capacity=5 * flags.batch_size)
  tn_label_bt = slim.one_hot_encoding(tn_label_bt, tn_dataset.num_classes)
  print('images={} labels={}'.format(tn_image_bt.shape, tn_label_bt.shape))
  tn_logit_bt, tn_end_point_bt = tn_tch_network(tn_image_bt)
  print('logits={}'.format(tn_logit_bt.shape))
  tf.losses.softmax_cross_entropy(tn_logit_bt, tn_label_bt)

  global_step = tf.Variable(0, trainable=False)
  learning_rate = utils.configure_learning_rate(
      flags,
      global_step,
      tn_dataset.num_samples,
      'tch')
  optimizer = utils.configure_optimizer(flags, learning_rate)

  losses = tf.get_collection(tf.GraphKeys.LOSSES)
  print('#loss=%d' % (len(losses)))
  losses.extend(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
  print('#loss=%d' % (len(losses)))
  total_loss = tf.add_n(losses, name='total_loss')
  update = optimizer.minimize(total_loss, global_step=global_step)

  tf.summary.scalar(learning_rate.name, learning_rate)
  tf.summary.scalar(total_loss.name, total_loss)
  summary_op = tf.summary.merge_all()
  init_op = tf.global_variables_initializer()

  num_batch = int(flags.num_epoch * tn_dataset.num_samples / flags.batch_size)
  start = time.time()
  with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    with slim.queues.QueueRunners(sess):
      for batch in range(num_batch):
        _, summary = sess.run([update, summary_op])
        writer.add_summary(summary, batch)

if __name__ == '__main__':
  tf.app.run()







