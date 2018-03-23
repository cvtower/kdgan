from kdgan import config
import data_utils

import keras
import time
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, InputLayer, MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2

batch_size = 128

cifar = data_utils.CIFAR()

image_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
hard_label_ph = tf.placeholder(tf.float32, shape=(None, 10))
isum = tf.reduce_sum(image_ph)
lsum = tf.reduce_sum(hard_label_ph)

sess = tf.Session()
with sess.as_default():
  start_time = time.time()
  for tn_batch in range(4000):
    tn_image_np, tn_label_np = cifar.train.next_batch(batch_size)
    feed_dict = {
      image_ph:tn_image_np,
      hard_label_ph:tn_label_np,
    }
    res = sess.run([isum, lsum], feed_dict=feed_dict)
  end_time = time.time()
  print('%.4fs' % (end_time - start_time))
