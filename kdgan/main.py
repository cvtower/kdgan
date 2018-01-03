from kdgan import config, utils
from kdgan.gen_model import GEN

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

from PIL import Image

tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
tf.app.flags.DEFINE_float('l2_coefficient', 0.00004, '')
FLAGS = tf.app.flags.FLAGS

batch_size_t = 32
batch_size_v = 100

id_to_label = utils.load_id_to_label()
num_label = len(id_to_label)

def check_tfrecord(bt_list, batch_size):
    id_to_label = utils.load_id_to_label()
    id_to_token = utils.load_id_to_token()

    user_bt, image_bt, text_bt, label_bt, image_file_bt = bt_list
    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            for t in range(3):
                user_np, image_np, text_np, label_np, image_file_np = sess.run(
                        [user_bt, image_bt, text_bt, label_bt, image_file_bt])
                print(user_np.shape, image_np.shape, text_np.shape, label_np.shape, image_file_np.shape)
                for b in range(batch_size):
                    print(user_np[b])
                    num_token = text_np[b].shape[0]
                    # tokens = [id_to_token[text_np[b, i]] for i in range(num_token)]
                    # print(tokens)
                    print(text_np[b])
        #             label_ids = [i for i, l in enumerate(label_np) if l != 0]
        #             labels = [id_to_label[label_id] for label_id in label_ids]
        #             print(labels)
        #             print(image_file_np)
        #             print('{0}\n{0}'.format('#'*80))
                    input()

def main(_):
    print('#label={}'.format(num_label))
    gen_t = GEN(FLAGS.model_name,
            num_label=num_label,
            l2_coefficient=FLAGS.l2_coefficient,
            is_training=True,
            preprocessing_name=FLAGS.preprocessing_name)
    gen_v = GEN(FLAGS.model_name,
            num_label=num_label,
            l2_coefficient=FLAGS.l2_coefficient,
            is_training=False,
            preprocessing_name=FLAGS.preprocessing_name)

    ts_list_t = utils.decode_tfrecord(config.train_tfrecord, shuffle=True)
    ts_list_v = utils.decode_tfrecord(config.valid_tfrecord, shuffle=False)
    bt_list_t = utils.generate_batch(gen_t, ts_list_t, batch_size_t)
    bt_list_v = utils.generate_batch(gen_v, ts_list_v, batch_size_v)
    check_tfrecord(bt_list_t, batch_size_t)
    check_tfrecord(bt_list_v, batch_size_v)

    _, image_bt_t, text_bt_t, label_bt_t, image_file_bt_t = bt_list_t
    _, image_bt_v, text_bt_v, label_bt_v, image_file_bt_v = bt_list_v



if __name__ == '__main__':
    tf.app.run()