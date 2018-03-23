from kdgan import config
import data_utils

import keras
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, InputLayer, MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2

num_epoch = 200
batch_size = 128
weight_decay  = 0.0001

cifar = data_utils.CIFAR()
tn_data_size = cifar.train.num_examples
tn_num_batch = int(num_epoch * tn_data_size / batch_size)
print('train #data=%d #batch=%d' % (tn_data_size, tn_num_batch))
eval_interval = int(max(tn_data_size / batch_size, 1.0))


with tf.Session() as sess:
  K.set_session(sess)

  image_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
  hard_label_ph = tf.placeholder(tf.float32, shape=(None, 10))

  model = Sequential()
  model.add(InputLayer(input_tensor=image_ph, input_shape=(None, 32, 32, 3)))
  model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', input_shape=(32,32,3)))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal'))
  model.add(MaxPooling2D((2, 2), strides=(2, 2)))
  model.add(Flatten())
  model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal'))
  model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal'))
  model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal'))

  labels = model.output
  print('label', type(labels), labels.shape)
  accuracy = tf.reduce_mean(categorical_accuracy(hard_label_ph, labels))

  reg_losses = model.losses
  for reg_loss in reg_losses:
    print('reg_loss', type(reg_loss), reg_loss.shape)

  hard_loss = tf.reduce_mean(categorical_crossentropy(labels, hard_label_ph))
  print('hard_loss', type(hard_loss), hard_loss.shape)

  pre_losses = [hard_loss]
  pre_losses.extend(reg_losses)

  pre_loss = tf.add_n(pre_losses)
  pre_update = tf.train.GradientDescentOptimizer(0.1).minimize(pre_loss)

  init_op = tf.global_variables_initializer()
  sess.run(init_op)
  for tn_batch in range(tn_num_batch):
    tn_image_np, tn_label_np = cifar.train.next_batch(batch_size)
    feed_dict = {
      image_ph:tn_image_np,
      hard_label_ph:tn_label_np,
      K.learning_phase(): 1,
    }
    res = sess.run(pre_update, feed_dict=feed_dict)

    if ((tn_batch + 1) % eval_interval != 0) and (tn_batch != (tn_num_batch - 1)):
      continue

    feed_dict = {
      image_ph:tn_image_np,
      hard_label_ph:tn_label_np,
      K.learning_phase(): 0,
    }
    acc = sess.run(accuracy, feed_dict=feed_dict)
    print('#batch=%d acc=%.4f' % (tn_batch, acc))

    # feed_dict = {
    #   image_ph:cifar.valid.images,
    #   hard_label_ph:cifar.valid.labels,
    #   K.learning_phase(): 0,
    # }
    # acc = sess.run(accuracy, feed_dict=feed_dict)
    # print('#batch=%d acc=%.4f' % (tn_batch, acc))
