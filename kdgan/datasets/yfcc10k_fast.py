from kdgan import config, utils

import os

import numpy as np
import tensorflow as tf

from nets import nets_factory
from os import path
from tensorflow.contrib import slim

from datasets import dataset_utils
from datasets.download_and_convert_flowers import ImageReader
from preprocessing import preprocessing_factory

from PIL import Image

tf.app.flags.DEFINE_boolean('dev', False, '')
tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
tf.app.flags.DEFINE_string('checkpoint_path', None, '')
tf.app.flags.DEFINE_string('end_point', None, '')

flags = tf.app.flags.FLAGS

FIELD_SEPERATOR = '\t'
POST_INDEX = 0
USER_INDEX = 1
IMAGE_INDEX = 2
TEXT_INDEX = 3
DESC_INDEX = 4
LABEL_INDEX = -1

num_classes = 1000
network_fn_t = nets_factory.get_network_fn(flags.model_name,
        num_classes=num_classes,
        is_training=True)
network_fn_v = nets_factory.get_network_fn(flags.model_name,
        num_classes=num_classes,
        is_training=False)
image_size_t = network_fn_t.default_image_size
image_size_v = network_fn_v.default_image_size
assert image_size_t==image_size_v
image_size = int((image_size_t + image_size_v) / 2)

image_ph = tf.placeholder(tf.float32,
        shape=(None, None, config.channels))
preprocessing_t = preprocessing_factory.get_preprocessing(flags.preprocessing_name,
        is_training=True)
preprocessing_v = preprocessing_factory.get_preprocessing(flags.preprocessing_name,
        is_training=False)
image_ts_t = tf.expand_dims(preprocessing_t(image_ph, image_size, image_size),
        axis=0)
image_ts_v = tf.expand_dims(preprocessing_v(image_ph, image_size, image_size),
        axis=0)
_, end_points_t = network_fn_t(image_ts_t)
for variable in slim.get_model_variables():
    num_params = 1
    for dim in variable.shape:
        num_params *= dim.value
    print('{} #param={}'.format(variable.name, num_params))
for key, tensor in end_points_t.items():
    print(key)
end_point_t = tf.squeeze(end_points_t[flags.end_point])
print(end_point_t.shape, end_point_t.dtype)

if flags.dev:
    exit()

variables_to_restore = slim.get_variables_to_restore()
# for variable in variables_to_restore:
#     print(variable.name, variable.shape)
init_fn = slim.assign_from_checkpoint_fn(flags.checkpoint_path, variables_to_restore)

def create_if_nonexist(outdir):
    if not path.exists(outdir):
        os.makedirs(outdir)

def build_example(user, image, text, label, file):
    return tf.train.Example(features=tf.train.Features(feature={
        config.user_key:dataset_utils.bytes_feature(user),
        config.image_key:dataset_utils.float_feature(image),
        config.text_key:dataset_utils.int64_feature(text),
        config.label_key:dataset_utils.int64_feature(label),
        config.file_key:dataset_utils.bytes_feature(file),
    }))

def create_tfrecord(infile, is_training=False):
    create_if_nonexist(config.prerecord_dir)

    num_epoch = 2
    if not is_training:
        num_epoch = 1

    fields = path.basename(infile).split('.')
    dataset, version = fields[0], fields[1]
    filename = '{0}_{1}_{2:03d}.{3}.tfrecord'
    filepath = path.join(config.prerecord_dir, filename)

    user_list = []
    file_list = []
    text_list = []
    label_list = []
    fin = open(infile)
    while True:
        line = fin.readline()
        if not line:
            break
        fields = line.strip().split(FIELD_SEPERATOR)
        user = fields[USER_INDEX]
        image = fields[IMAGE_INDEX]
        file = path.join(config.image_data_dir, '%s.jpg' % image)
        tokens = fields[TEXT_INDEX].split()
        labels = fields[LABEL_INDEX].split()
        user_list.append(user)
        file_list.append(file)
        text_list.append(tokens)
        label_list.append(labels)
    fin.close()

    label_to_id = utils.load_label_to_id()
    num_label = len(label_to_id)
    print('#label={}'.format(num_label))
    token_to_id = utils.load_token_to_id()
    unk_token_id = token_to_id[config.unk_token]
    vocab_size = len(token_to_id)
    print('#vocab={}'.format(vocab_size))

    reader = ImageReader()
    with tf.Session() as sess:
        init_fn(sess)
        for epoch in range(num_epoch):
            count = 0
            tfrecord_file = filepath.format(dataset, flags.model_name, epoch, version)
            with tf.python_io.TFRecordWriter(tfrecord_file) as fout:
                for user, file, text, labels in zip(user_list, file_list, text_list, label_list):
                    user = bytes(user, encoding='utf-8')
                    
                    image_np = np.array(Image.open(file))
                    # print(type(image_np), image_np.shape)
                    feed_dict = {image_ph:image_np}
                    image_t, = sess.run([end_point_t], feed_dict)
                    # print(type(image), image.shape)
                    image_t = image_t.tolist()
                    # print(type(image_t), len(image_t), image_t)

                    text = [token_to_id.get(token, unk_token_id) for token in text]

                    label_ids = [label_to_id[label] for label in labels]
                    label_vec = np.zeros((num_label,), dtype=np.int64)
                    label_vec[label_ids] = 1
                    label = label_vec.tolist()

                    file = bytes(file, encoding='utf-8')
                    # print(file)

                    example = build_example(user, image_t, text, label, file)
                    fout.write(example.SerializeToString())
                    count += 1
                    if (count % 200) == 0:
                        print('count={}'.format(count))

def main(_):
    create_tfrecord(config.train_file, is_training=True)
    create_tfrecord(config.valid_file, is_training=False)

if __name__ == '__main__':
    tf.app.run()