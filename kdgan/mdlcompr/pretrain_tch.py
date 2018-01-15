from kdgan import config
from kdgan import metric
from kdgan import utils
from tch_model import TCH
import data_utils

from os import path
from tensorflow.contrib import slim
import time
import numpy as np
import tensorflow as tf

# dataset
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_integer('image_size', 28, '')
tf.app.flags.DEFINE_integer('channels', 1, '')
# model
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, '')
tf.app.flags.DEFINE_string('checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('save_path', None, '')
tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
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

tn_dataset = data_utils.get_dataset(flags, is_training=True)
vd_dataset = data_utils.get_dataset(flags, is_training=False)
tn_image_bt, tn_label_bt = data_utils.generate_batch(flags, tn_dataset, is_training=True)
vd_image_bt, vd_label_bt = data_utils.generate_batch(flags, vd_dataset, is_training=False)

tn_tch = TCH(flags, tn_dataset, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_tch = TCH(flags, vd_dataset, is_training=False)

tf.summary.scalar(tn_tch.learning_rate.name, tn_tch.learning_rate)
tf.summary.scalar(tn_tch.pre_loss.name, tn_tch.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

def main(_):
  tch_ckpt = tf.train.latest_checkpoint(flags.checkpoint_dir)
  # print('tch ckpt=%s' % (tch_ckpt))
  if tch_ckpt != None:
    print('todo init from tch ckpt')
    exit()
  utils.create_if_nonexist(flags.checkpoint_dir)

  for variable in tf.trainable_variables():
    num_params = 1
    for dim in variable.shape:
      num_params *= dim.value
    print('%-50s (%d params)' % (variable.name, num_params))

  tn_num_batch = int(flags.num_epoch * tn_dataset.num_samples / flags.batch_size)
  vd_num_batch = int(vd_dataset.num_samples / config.valid_batch_size)
  print('tn #batch=%d vd #batch=%d' % (tn_num_batch, vd_num_batch))
  eval_interval = int(tn_dataset.num_samples / flags.batch_size)
  print('ev #interval=%d' % (eval_interval))
  start = time.time()
  best_acc_v = -np.inf
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    for tn_batch in range(tn_num_batch):
      tn_image_np, tn_label_np = sess.run([tn_image_bt, tn_label_bt])
      # print('tn image shape={} dtype={}'.format(tn_image_np.shape, tn_image_np.dtype))
      # print('tn label shape={} dtype={}'.format(tn_label_np.shape, tn_label_np.dtype))
      feed_dict = {tn_tch.image_ph:tn_image_np, tn_tch.hard_label_ph:tn_label_np}
      _, summary = sess.run([tn_tch.pre_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, tn_batch)

      if (tn_batch + 1) % eval_interval != 0:
        continue
      acc_v = []
      for vd_batch in range(vd_num_batch):
        vd_image_np, vd_label_np = sess.run([vd_image_bt, vd_label_bt])
        feed_dict = {vd_tch.image_ph:vd_image_np}
        predictions, = sess.run([vd_tch.predictions], feed_dict=feed_dict)
        acc_v.append(metric.compute_acc(predictions, vd_label_np))
      acc_v = np.mean(acc_v)
      tot_time = time.time() - start
      print('#%08d hit=%.4f %06ds' % (tn_batch, acc_v, int(tot_time)))

      if acc_v < best_acc_v:
        continue
      best_acc_v = acc_v
      global_step, = sess.run([tn_tch.global_step])
      tn_tch.saver.save(utils.get_session(sess), flags.save_path, global_step=global_step)
  print('best acc=%.4f' % (best_acc_v))

if __name__ == '__main__':
  tf.app.run()

def test():
  def parser(record):
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
      'image/class/label': tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], dtype=tf.int64)),
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    return parsed['image/encoded'], parsed['image/class/label']
  vd_label_cn = {}
  valid_data_size = 10000
  num_batch = int(valid_data_size / config.valid_batch_size)
  valid_filepath = path.join(flags.dataset_dir, 'mnist_valid.tfrecord' )
  dataset = tf.data.TFRecordDataset([valid_filepath])
  dataset = dataset.map(parser)
  dataset = dataset.batch(config.valid_batch_size)
  iterator = dataset.make_one_shot_iterator()
  image_bt, label_bt = iterator.get_next()
  with tf.train.MonitoredTrainingSession() as sess:
    while not sess.should_stop():
      image_np, label_np = sess.run([image_bt, label_bt])
      for label in label_np:
        label = int(label)
        vd_label_cn[label] = vd_label_cn.get(label, 0) + 1
  for label in range(10):
    print('%d vd=%d' % (label, vd_label_cn.get(label, 0)))


  valid_data_size = vd_dataset.num_samples
  num_batch = int(valid_data_size / config.valid_batch_size)
  with tf.train.MonitoredTrainingSession() as sess:
    tn_label_cn, vd_label_cn = {}, {}
    for i in range(num_batch):
      tn_image_np, tn_label_np = sess.run([tn_image_bt, tn_label_bt])
      vd_image_np, vd_label_np = sess.run([vd_image_bt, vd_label_bt])
      for tn_label in np.argmax(tn_label_np, axis=1):
        tn_label_cn[tn_label] = tn_label_cn.get(tn_label, 0) + 1
      for vd_label in np.argmax(vd_label_np, axis=1):
        vd_label_cn[vd_label] = vd_label_cn.get(vd_label, 0) + 1
  for label in range(10):
    print('%d tn=%d vd=%d' % (label, tn_label_cn[label], vd_label_cn[label]))





