from kdgan import config
from kdgan import utils
from flags import flags
from std_model import STD
from data_utils import CIFAR
import cifar10_utils

from keras import backend as K
from keras.initializers import Constant, TruncatedNormal
from keras.layers import Conv2D, Dense, Flatten, InputLayer, MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import Sequential
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2
import numpy as np
import tensorflow as tf
import math
import time

cifar = CIFAR(flags)
tn_num_batch = int(flags.num_epoch * flags.train_size / flags.batch_size)
print('#tn_batch=%d' % (tn_num_batch))
eval_interval = int(math.ceil(flags.train_size / flags.batch_size))

image_shape = (flags.batch_size, flags.image_size, flags.image_size, flags.channels)
image_ph = tf.placeholder(tf.float32, shape=image_shape)
hard_label_ph = tf.placeholder(tf.int32, shape=(flags.batch_size))

from keras.layers.core import Layer
class LRN(Layer):
  def __init__(self, n, k=1, alpha=0.0001, beta=0.75, **kwargs):
    self.n = n
    self.alpha = alpha
    self.k = k
    self.beta = beta
    super(LRN, self).__init__(**kwargs)

  def call(self, x, mask=None):
    b, ch, r, c = x.shape
    half_n = self.n // 2 # half the local region
    # orig keras code
    #input_sqr = T.sqr(x)  # square the input
    input_sqr = K.square(x) # square the input
    # orig keras code
    #extra_channels = T.alloc(0., b, ch + 2 * half_n, r,c)  # make an empty tensor with zero pads along channel dimension
    #input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input

    extra_channels = K.zeros((b, int(ch) + 2 * half_n, r, c))
    input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],input_sqr, extra_channels[:, half_n + int(ch):, :, :]],axis = 1)

    scale = self.k # offset for the scale
    norm_alpha = self.alpha / self.n # normalized alpha
    for i in range(self.n):
      scale += norm_alpha * input_sqr[:, i:i+int(ch), :, :]
    scale = scale ** self.beta
    x = x / scale
    return x

  def get_config(self):
    config = {"alpha": self.alpha,
              "k": self.k,
              "beta": self.beta,
              "n": self.n}
    base_config = super(LRN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

model = Sequential()
model.add(InputLayer(input_tensor=image_ph, input_shape=image_shape))
model.add(Conv2D(64, (5, 5), 
    padding='same', 
    activation='relu',
    kernel_initializer=TruncatedNormal(stddev=0.05),
    kernel_regularizer=None,
    bias_initializer=Constant(value=0)))
model.add(MaxPooling2D((3, 3),
    strides=(2, 2)))
model.add(LRN(4,
    k=1.0,
    alpha=0.001 / 9.0,
    beta=0.75))
model.add(Conv2D(64, (5, 5),
    padding='same',
    activation='relu',
    kernel_initializer=TruncatedNormal(stddev=0.05),
    kernel_regularizer=None,
    bias_initializer=Constant(value=0.1)))
model.add(MaxPooling2D((3, 3),
    strides=(2, 2)))
model.add(Flatten())
model.add(Dense(384,
    activation='relu',
    kernel_initializer=TruncatedNormal(stddev=0.04),
    kernel_regularizer=l2(flags.tch_weight_decay),
    bias_initializer=Constant(value=0.1)))
model.add(Dense(192,
    activation='relu',
    kernel_initializer=TruncatedNormal(stddev=0.04),
    kernel_regularizer=l2(flags.tch_weight_decay),
    bias_initializer=Constant(value=0.1)))
model.add(Dense(10,
    activation=None,
    kernel_initializer=TruncatedNormal(stddev=1/192.0),
    kernel_regularizer=None,
    bias_initializer=Constant(value=0.0)))
logits = model.output

hard_loss = cifar10_utils.loss(logits, hard_label_ph)
regularization_losses = model.losses
pre_losses = [hard_loss]
pre_losses.extend(regularization_losses)
pre_loss = tf.add_n(pre_losses)

top_k_op = tf.nn.in_top_k(logits, hard_label_ph, 1)
accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32))

global_step = tf.Variable(0, trainable=False)
pre_train = cifar10_utils.get_train_op(flags, pre_loss, global_step)

init_op = tf.global_variables_initializer()

# for variable in tf.trainable_variables():
#   print('variable.name', variable.name)

def main(argv=None):
  bst_acc = 0.0
  with tf.Session() as sess:
    sess.run(init_op)
    start_time = time.time()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)        
    try:
      for tn_batch in range(tn_num_batch):
        tn_image_np, tn_label_np = cifar.next_batch(sess)
        feed_dict = {
          image_ph:tn_image_np,
          hard_label_ph:tn_label_np,
          K.learning_phase(): 1,
        }
        sess.run(pre_train, feed_dict=feed_dict)
        if (tn_batch + 1) % eval_interval != 0 and (tn_batch + 1) != tn_num_batch:
          continue
        acc = cifar.evaluate(sess, image_ph, hard_label_ph, accuracy)
        bst_acc = max(acc, bst_acc)

        end_time = time.time()
        duration = end_time - start_time
        avg_time = duration / (tn_batch + 1)
        print('#batch=%d acc=%.4f time=%.4fs/batch est=%.4fh' % 
            (tn_batch + 1, bst_acc, avg_time, avg_time * tn_num_batch / 3600))

        if acc < bst_acc:
          continue
        # tn_std.saver.save(utils.get_session(sess), flags.std_model_ckpt)
    except tf.errors.OutOfRangeError as e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads) 
  print('final=%.4f' % (bst_acc))

if __name__ == '__main__':
  tf.app.run()










