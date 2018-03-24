from kdgan import config
import cifar10_utils

from datetime import datetime
import tensorflow as tf
import time

FLAGS = tf.app.flags.FLAGS

def speed():
  with tf.device('/cpu:0'):
    image_ts, label_ts = cifar10_utils.distorted_inputs()
    isum_alp = tf.reduce_sum(image_ts)
    lsum_alp = tf.reduce_sum(label_ts)

    image_ph = tf.placeholder(tf.float32, shape=(None, 24, 24, 3))
    label_ph = tf.placeholder(tf.int32, shape=(None,))
    isum_bet = tf.reduce_sum(image_ph)
    lsum_bet = tf.reduce_sum(label_ph)

  with tf.train.MonitoredTrainingSession() as sess:
    start_time = time.time()
    for tn_batch in range(10000):
      isum_np, lsum_np = sess.run([isum_alp, lsum_alp])
      if (tn_batch + 1) % 2000 == 0:
        print('#alp=%d' % (tn_batch + 1))
    end_time = time.time()
    duration = end_time - start_time
    print('alp=%.4fs' % (duration))

    start_time = time.time()
    for tn_batch in range(10000):
      image_np, label_np = sess.run([image_ts, label_ts])
      feed_dict = {image_ph:image_np, label_ph:label_np}
      isum_np, lsum_np = sess.run([isum_bet, lsum_bet], feed_dict=feed_dict)
      if (tn_batch + 1) % 2000 == 0:
        print('#bet=%d' % (tn_batch + 1))
    end_time = time.time()
    duration = end_time - start_time
    print('bet=%.4fs' % (duration))

def train():
  with tf.device('/cpu:0'):
    image_ts, label_ts = cifar10_utils.distorted_inputs()

    image_shape = (None, flags.image_size, flags.image_size, flags.channels)
    image_ph = tf.placeholder(tf.float32, shape=image_shape)
    label_ph = tf.placeholder(tf.int32, shape=(None,))

  logit_ts = cifar10_utils.inference(image_ph)
  loss_ts = cifar10_utils.loss(logit_ts, label_ph)
  global_step = tf.Variable(0, trainable=False)
  train_op = cifar10_utils.train(loss_ts, global_step)

  with tf.train.MonitoredTrainingSession() as sess:
    for tn_batch in range(10000):
      image_np, label_np = sess.run([image_ts, label_ts])
      feed_dict = {image_ph:image_np, label_ph:label_np}
      _, loss = sess.run([train_op, loss_ts], feed_dict=feed_dict)
      print('#batch=%d loss=%.4f' % (tn_batch, loss))

def main(argv=None):
  cifar10_utils.maybe_download_and_extract()
  train()

if __name__ == '__main__':
  tf.app.run()










