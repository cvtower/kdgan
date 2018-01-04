from kdgan import config

import numpy as np
import tensorflow as tf

from preprocessing import preprocessing_factory
from tensorflow.contrib import slim

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

def load_label_to_id():
    label_to_id = load_sth_to_id(config.label_file)
    return label_to_id

def load_token_to_id():
    vocab_to_id = load_sth_to_id(config.vocab_file)
    return vocab_to_id

def load_id_to_sth(infile):
    with open(infile) as fin:
        sth_list = [sth.strip() for sth in fin.readlines()]
    id_to_sth = dict(zip(range(len(sth_list)), sth_list))
    return id_to_sth

def load_id_to_label():
    id_to_label = load_id_to_sth(config.label_file)
    return id_to_label

def load_id_to_token():
    id_to_vocab = load_id_to_sth(config.vocab_file)
    return id_to_vocab

def count_data_size(infile):
    with open(infile) as fin:
        data = [line.strip() for line in fin.readlines()]
    data_size = len(data)
    return data_size

def decode_tfrecord(flags, tfrecord_file, shuffle=True):
    Tensor = slim.tfexample_decoder.Tensor
    TFExampleDecoder = slim.tfexample_decoder.TFExampleDecoder
    Dataset = slim.dataset.Dataset
    DatasetDataProvider = slim.dataset_data_provider.DatasetDataProvider

    data_sources = [tfrecord_file]
    num_label = config.num_label
    token_to_id = load_token_to_id()
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
    user_bt, image_bt, text_bt, label_bt, file_bt = tf.train.batch(
            [user_ts, image_ts, text_ts, label_ts, file_ts], 
            batch_size=batch_size,
            dynamic_pad=True,
            num_threads=config.num_threads)
    return user_bt, image_bt, text_bt, label_bt, file_bt






