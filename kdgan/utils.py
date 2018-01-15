from kdgan import config
from kdgan import metric

import os
import numpy as np
import tensorflow as tf
from os import path
from preprocessing import preprocessing_factory
from tensorflow.contrib import slim

def create_if_nonexist(outdir):
  if not path.exists(outdir):
    os.makedirs(outdir)

def skip_if_exist(infile):
  skip = False
  if path.isfile(infile):
    skip = True
  return skip

def save_collection(coll, outfile):
  with open(outfile, 'w') as fout:
    for elem in coll:
      fout.write('%s\n' % elem)

def load_collection(infile):
  with open(infile) as fin:
    coll = [elem.strip() for elem in fin.readlines()]
  return coll

def load_sth_to_id(infile):
  with open(infile) as fin:
    sth_list = [sth.strip() for sth in fin.readlines()]
  sth_to_id = dict(zip(sth_list, range(len(sth_list))))
  return sth_to_id

def load_label_to_id(dataset):
  label_file = get_label_file()
  label_to_id = load_sth_to_id(label_file)
  return label_to_id

def load_token_to_id(dataset):
  vocab_file = get_vocab_file(dataset)
  vocab_to_id = load_sth_to_id(vocab_file)
  return vocab_to_id

def load_id_to_sth(infile):
  with open(infile) as fin:
    sth_list = [sth.strip() for sth in fin.readlines()]
  id_to_sth = dict(zip(range(len(sth_list)), sth_list))
  return id_to_sth

def load_id_to_label():
  label_file = get_label_file()
  label_to_id = load_sth_to_id(label_file)
  id_to_label = load_id_to_sth(label_file)
  return id_to_label

def load_id_to_token():
  vocab_file = get_vocab_file(dataset)
  vocab_to_id = load_sth_to_id(vocab_file)
  id_to_vocab = load_id_to_sth(vocab_file)
  return id_to_vocab

def count_data_size(infile):
  with open(infile) as fin:
    data = [line.strip() for line in fin.readlines()]
  data_size = len(data)
  return data_size

def get_data_sources(flags, is_training=True, single_source=False):
  precomputed_dir = get_precomputed_dir(flags.dataset)
  for (dirpath, dirnames, filenames) in os.walk(precomputed_dir):
    break
  marker = 'train'
  if not is_training:
    marker = 'valid'
  data_sources = []
  for filename in filenames:
    if filename.find(marker) < 0:
      continue
    if filename.find(flags.model_name) < 0:
      continue
    if single_source and (filename.find('000') < 0):
      continue
    filepath = path.join(precomputed_dir, filename)
    data_sources.append(filepath)
  return data_sources

def decode_tfrecord(flags, data_sources, shuffle=True):
  Tensor = slim.tfexample_decoder.Tensor
  TFExampleDecoder = slim.tfexample_decoder.TFExampleDecoder
  Dataset = slim.dataset.Dataset
  DatasetDataProvider = slim.dataset_data_provider.DatasetDataProvider

  num_label = config.num_label
  token_to_id = load_token_to_id(flags.dataset)
  unk_token_id = token_to_id[config.unk_token]
  reader = tf.TFRecordReader
  keys_to_features = {
    config.user_key:tf.FixedLenFeature((), tf.string),
    config.image_key:tf.FixedLenFeature([flags.feature_size], tf.float32),
    config.text_key:tf.VarLenFeature(dtype=tf.int64),
    config.label_key:tf.FixedLenFeature([num_label], tf.int64),
    config.file_key:tf.FixedLenFeature((), tf.string)
  }
  items_to_handlers = {
    'user':Tensor(config.user_key),
    'image':Tensor(config.image_key),
    'text':Tensor(config.text_key, default_value=unk_token_id),
    'label':Tensor(config.label_key),
    'file':Tensor(config.file_key),
  }
  decoder = TFExampleDecoder(keys_to_features, items_to_handlers)
  num_samples = np.inf
  items_to_descriptions = {
    'user':'',
    'image':'',
    'text':'',
    'label':'',
    'file':'',
  }
  dataset = Dataset(
    data_sources=data_sources,
    reader=reader,
    decoder=decoder,
    num_samples=num_samples,
    items_to_descriptions=items_to_descriptions,
  )
  provider = DatasetDataProvider(dataset, shuffle=shuffle)
  ts_list = provider.get(['user', 'image', 'text', 'label', 'file'])
  return ts_list

def generate_batch(ts_list, batch_size):
  user_ts, image_ts, text_ts, label_ts, file_ts = ts_list
  label_ts = tf.divide(label_ts, tf.reduce_sum(label_ts))
  user_bt, image_bt, text_bt, label_bt, file_bt = tf.train.batch(
      [user_ts, image_ts, text_ts, label_ts, file_ts], 
      batch_size=batch_size,
      dynamic_pad=True,
      num_threads=config.num_threads)
  return user_bt, image_bt, text_bt, label_bt, file_bt

def evaluate(flags, sess, gen_v, bt_list_v):
  valid_data_size = get_valid_data_size(flags.dataset)
  num_batch_v = int(valid_data_size / config.valid_batch_size)
  # print('vd:\t#batch=%d\n' % num_batch_v)
  user_bt_v, image_bt_v, text_bt_v, label_bt_v, file_bt_v = bt_list_v
  image_hit_v = []
  for batch_v in range(num_batch_v):
    image_np_v, label_np_v = sess.run([image_bt_v, label_bt_v])
    feed_dict = {gen_v.image_ph:image_np_v}
    
    image_logit_v, = sess.run([gen_v.logits], feed_dict=feed_dict)
    image_hit_bt = metric.compute_hit(image_logit_v, label_np_v, flags.cutoff)
    image_hit_v.append(image_hit_bt)
  image_hit_v = np.mean(image_hit_v)
  return image_hit_v

def get_train_data_size(dataset):
  train_data_sizes = {
    'yfcc10k':9500,
    'yfcc20k':19000,
  }
  train_data_size = train_data_sizes[dataset]
  return train_data_size

def get_valid_data_size(dataset):
  valid_data_sizes = {
    'yfcc10k':500,
    'yfcc20k':1000,
  }
  valid_data_size = valid_data_sizes[dataset]
  return valid_data_size

def get_vocab_size(dataset):
  vocab_sizes = {
    'yfcc10k':6936,
    'yfcc20k':8756,
  }
  vocab_size = vocab_sizes[dataset]
  return vocab_size

def get_dataset_dir(dataset):
  dataset_dir = path.join(config.yfcc_dir, dataset)
  return dataset_dir

def get_precomputed_dir(dataset):
  dataset_dir = get_dataset_dir(dataset)
  precomputed_dir = path.join(dataset_dir, 'Precomputed')
  return precomputed_dir

def get_vocab_file(dataset):
  dataset_dir = get_dataset_dir(dataset)
  vocab_file = path.join(dataset_dir, '%s.vocab' % dataset)
  return vocab_file

def get_label_file(dataset):
  dataset_dir = get_dataset_dir(dataset)
  label_file = path.join(dataset_dir, '%s.label' % dataset)
  return label_file

def get_lr(flags, global_step, train_data_size, scope_name):
  decay_steps = int(train_data_size / flags.batch_size * flags.num_epochs_per_decay)
  if flags.learning_rate_decay_type == 'exponential':
    name = '%s_exponential_decay_learning_rate' % scope_name
    learning_rate = tf.train.exponential_decay(flags.learning_rate,
        global_step, decay_steps, flags.learning_rate_decay_factor,
        staircase=True,
        name=name)
  elif flags.learning_rate_decay_type == 'fixed':
    name = '%s_fixed_learning_rate' % scope_name
    learning_rate = tf.constant(flags.learning_rate,
        name=name)
  elif flags.learning_rate_decay_type == 'polynomial':
    name = '%s_polynomial_decay_learning_rate' % scope_name
    learning_rate = tf.train.polynomial_decay(flags.learning_rate,
        global_step, decay_steps, flags.end_learning_rate,
        power=1.0,
        cycle=False,
        name=name)
  else:
    raise ValueError('bad learning rate decay type %s', flags.learning_rate_decay_type)
  return learning_rate

def get_opt(flags, learning_rate):
  if flags.optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
        learning_rate,
        beta1=flags.adam_beta1,
        beta2=flags.adam_beta2,
        epsilon=flags.opt_epsilon)
  elif flags.optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate,
        decay=flags.rmsprop_decay,
        momentum=flags.rmsprop_momentum,
        epsilon=flags.opt_epsilon)
  elif flags.optimizer == 'sgd':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  else:
    raise ValueError('bad optimizer %s', flags.optimizer)
  return optimizer

def gan_dis_sample(flags, label_dat, label_gen):
  # print('{0} {1:.2f}'.format(label_dat.shape, label_dat.sum()))
  # print('{0} {1:.2f}'.format(label_gen.shape, label_gen.sum()))
  sample_np, label_np = [], []
  for batch, (label_d, label_g) in enumerate(zip(label_dat, label_gen)):
    num_sample = np.count_nonzero(label_d)
    # print(batch, label_d.shape, label_g.shape, num_sample)
    num_positive = num_sample * flags.num_positive
    sample_d = np.random.choice(config.num_label, num_positive, p=label_d)
    for sample in sample_d:
      # print(batch, sample, 1.0)
      sample_np.append((batch, sample))
      label_np.append(1.0)
    num_negative = num_sample * flags.num_negative
    sample_g = np.random.choice(config.num_label, num_negative, p=label_g)
    for sample in sample_g:
      sample_np.append((batch, sample))
      label_np.append(0.0)
  sample_np = np.asarray(sample_np)
  label_np = np.asarray(label_np)
  # for sample, label in zip(sample_np, label_np):
  #   print(sample, label)
  return sample_np, label_np

def kdgan_dis_sample(flags, label_dat, label_gen, label_tch):
  label_zip = zip(label_dat, label_gen, label_tch)
  sample_np, label_np = [], []
  for batch, (label_d, label_g, label_t) in enumerate(label_zip):
    num_sample = np.count_nonzero(label_d)
    # print('batch=%d #sample=%d' % (batch, num_sample))
    # print('%s sum=%.2f' % (label_d.shape, label_d.sum()))
    # print('%s sum=%.2f' % (label_g.shape, label_g.sum()))
    # print('%s sum=%.2f' % (label_t.shape, label_t.sum()))
    num_positive = num_sample * flags.num_positive
    sample_d = np.random.choice(config.num_label, num_positive, p=label_d)
    for sample in sample_d:
      sample_np.append((batch, sample))
      label_np.append(1.0)

    num_negative = num_sample * flags.num_negative
    sample_g = np.random.choice(config.num_label, num_negative, p=label_g)
    for sample in sample_g:
      sample_np.append((batch, sample))
      label_np.append(0.0)
    sample_t = np.random.choice(config.num_label, num_negative, p=label_t)
    for sample in sample_t:
      sample_np.append((batch, sample))
      label_np.append(0.0)
  sample_np = np.asarray(sample_np)
  label_np = np.asarray(label_np)
  # for sample, label in zip(sample_np, label_np):
  #   print(sample, label)
  return sample_np, label_np

def generate_label(flags, label_dat, label_gen):
  sample_np = []
  for batch, (label_d, label_g) in enumerate(zip(label_dat, label_gen)):
    num_sample = np.count_nonzero(label_d)
    num_sample *= (flags.num_positive + flags.num_negative)
    # print(num_sample, label_g.sum())
    # if abs(label_g.sum() - 1.0) > 0.001:
    #   print(label_g)
    #   exit()
    sample_g = np.random.choice(config.num_label, num_sample, p=label_g)
    for sample in sample_g:
      # if (sample < 0) or (sample > config.num_label - 1):
      #   print(sample_g)
      #   exit()
      sample_np.append((batch, sample))
  sample_np = np.asarray(sample_np)
  return sample_np

def get_session(sess):
  session = sess
  while type(session).__name__ != 'Session':
    session = session._sess
  return session

def get_latest_ckpt(checkpoint_dir):
  ckpt_file = tf.train.latest_checkpoint(checkpoint_dir)
  if ckpt_file == None:
    print('%s does not have any checkpoint' % (path.basename(checkpoint_dir)))
    exit()
  return ckpt_file




