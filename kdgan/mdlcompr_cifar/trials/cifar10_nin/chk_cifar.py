from kdgan import config
import data_utils

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.layers import InputLayer, Reshape
from keras.metrics import categorical_accuracy
from keras.objectives import categorical_crossentropy
from keras.regularizers import l2

batch_size = 128
weight_decay  = 0.0001

import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

image_ph = tf.placeholder(tf.float32, shape=(None, 32 * 32 * 3))
hard_label_ph = tf.placeholder(tf.float32, shape=(None, 10))

model = Sequential()
model.add(InputLayer(input_tensor=image_ph, input_shape=(None, 32 * 32 * 3)))
model.add(Reshape((32, 32, 3)))
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
# pre_update = tf.train.GradientDescentOptimizer(0.05).minimize(pre_loss)
pre_update = tf.train.GradientDescentOptimizer(0.05).minimize(hard_loss)

init_op = tf.global_variables_initializer()
sess.run(init_op)

from include.data import get_data_set
train_x, train_y, train_l = get_data_set()
test_x, test_y, test_l = get_data_set("test")
print(type(test_x), type(test_y), type(test_l))
# print(test_x.shape, test_y.shape, test_l.shape)
exit()

with sess.as_default():
  for tn_batch in range(10000):
    randidx = np.random.randint(len(train_x), size=batch_size)
    batch_xs = train_x[randidx]
    batch_ys = train_y[randidx]
    feed_dict = {
      image_ph:batch_xs,
      hard_label_ph:batch_ys,
      K.learning_phase(): 1,
    }
    # res = sess.run(pre_update, feed_dict=feed_dict)
    pre_update.run(feed_dict=feed_dict)

    if (tn_batch + 1) % 100 != 0:
      continue
    feed_dict = {
      image_ph:x_test,
      hard_label_ph:y_test,
      K.learning_phase(): 0,
    }
    acc = sess.run(accuracy, feed_dict=feed_dict)
    acc = accuracy.eval(feed_dict=feed_dict)
    print('#batch=%d acc=%.4f' % (tn_batch, acc))
