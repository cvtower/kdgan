from kdgan import config
from flags import flags
import cifar10_utils

from datetime import datetime
import tensorflow as tf
import time

cifar10_utils.maybe_download_and_extract()
with tf.device('/cpu:0'):
  tn_image_ts, tn_label_ts = cifar10_utils.distorted_inputs()
  

def main(argv=None):
  image_shape = (flags.batch_size, flags.image_size, flags.image_size, flags.channels)
  image_ph = tf.placeholder(tf.float32, shape=image_shape)
  label_ph = tf.placeholder(tf.int32, shape=(flags.batch_size,))

  # logit_ts = cifar10_utils.inference(image_ph)
  # loss_ts = cifar10_utils.loss(logit_ts, label_ph)
  # global_step = tf.Variable(0, trainable=False)
  # train_op = cifar10_utils.train(loss_ts, global_step)

  # start_time = time.time()
  # with tf.train.MonitoredTrainingSession() as sess:
  #   for tn_batch in range(100000):
  #     image_np, label_np = sess.run([image_ts, label_ts])
  #     feed_dict = {image_ph:image_np, label_ph:label_np}
  #     _, loss = sess.run([train_op, loss_ts], feed_dict=feed_dict)
  #     if (tn_batch + 1) % 10000 != 0:
  #       continue
  #     print('#batch=%d loss=%.4f' % (tn_batch, loss))
  # end_time = time.time()
  # duration = end_time - start_time
  # print('duration=%.4f' % (duration))

if __name__ == '__main__':
  tf.app.run()










