from kdgan import config
from kdgan import utils

from datasets import dataset_factory
from nets import nets_factory
from os import path
from preprocessing import preprocessing_factory
from tensorflow.contrib import slim
import tensorflow as tf

def get_dataset(flags, is_training=True):
  if is_training:
    split_name = 'train'
  else:
    split_name = 'valid'
  dataset_name = path.basename(flags.dataset_dir)
  dataset = dataset_factory.get_dataset(dataset_name, split_name, flags.dataset_dir)
  print('%s #dataset=%d' % (split_name, dataset.num_samples))
  return dataset

def generate_batch(flags, dataset, is_training=True):
  if is_training:
    batch_size = flags.batch_size
    shuffle = True
    ## cause not to traverse valid data
    num_readers = config.num_readers
  else:
    batch_size = config.valid_batch_size
    shuffle = False
    ## cause not to traverse valid data
    num_readers = 1
  print('#dataset=%d #batch=%d shuffle=%s' % (dataset.num_samples, batch_size, shuffle))

  preprocessing = preprocessing_factory.get_preprocessing(flags.preprocessing_name,
      is_training=is_training)

  provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
      num_readers=num_readers,
      common_queue_capacity=20 * batch_size,
      common_queue_min=10 * batch_size,
      shuffle=shuffle)

  [image_ts, label_ts] = provider.get(['image', 'label'])
  image_ts = preprocessing(image_ts, flags.image_size, flags.image_size)
  image_bt, label_bt = tf.train.batch(
      [image_ts, label_ts],
      num_threads=config.num_preprocessing_threads,
      capacity=5 * batch_size,
      batch_size=batch_size,)
  # label_bt = slim.one_hot_encoding(label_bt, dataset.num_classes)
  return image_bt, label_bt



