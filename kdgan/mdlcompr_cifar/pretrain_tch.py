from kdgan import config
from kdgan import utils
from flags import flags
from std_model import STD
from data_utils import CIFAR
import cifar10_utils

from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, InputLayer, MaxPooling2D
from keras.metrics import categorical_accuracy
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

model = Sequential()

model = Sequential()
model.add(InputLayer(input_tensor=image_ph, input_shape=image_shape))
model.add(Conv2D(6, (5, 5), 
    padding='valid', 
    activation='relu',
    kernel_initializer='he_normal',
    kernel_regularizer=l2(flags.tch_weight_decay),
    input_shape=(flags.image_size, flags.image_size, flags.channels)))
model.add(MaxPooling2D((2, 2),
    strides=(2, 2)))
model.add(Conv2D(16, (5, 5),
    padding='valid',
    activation='relu',
    kernel_initializer='he_normal',
    kernel_regularizer=l2(flags.tch_weight_decay)))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(120,
    activation='relu',
    kernel_initializer='he_normal',
    kernel_regularizer=l2(flags.tch_weight_decay)))
model.add(Dense(84,
    activation='relu',
    kernel_initializer='he_normal',
    kernel_regularizer=l2(flags.tch_weight_decay)))
# model.add(Dense(10,
#     activation='softmax',
#     kernel_initializer='he_normal',
#     kernel_regularizer=l2(flags.tch_weight_decay)))
logits = model.output


hard_loss = cifar10_utils.loss(logits, hard_label_ph)
reg_losses = model.losses
pre_losses = [hard_loss]
pre_losses.extend(reg_losses)
pre_loss = tf.add_n(pre_losses)

top_k_op = tf.nn.in_top_k(logits, hard_label_ph, 1)
accuracy = tf.reduce_mean(tf.cast(top_k_op, tf.float32))

global_step = tf.Variable(0, trainable=False)
pre_train = cifar10_utils.get_train_op(flags, pre_loss, global_step)

init_op = tf.global_variables_initializer()

def main(argv=None):
  bst_acc = 0.0
  with tf.Session() as sess:
    sess.run(init_op)
    start_time = time.time()
    
    from tensorflow.python.training import coordinator
    from tensorflow.python.training import queue_runner
    coord = coordinator.Coordinator(clean_stop_exception_types=[])
    queue_runner.start_queue_runners(sess=sess, coord=coord)
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
      tn_std.saver.save(utils.get_session(sess), flags.std_model_ckpt)
  print('final=%.4f' % (bst_acc))

if __name__ == '__main__':
  tf.app.run()










