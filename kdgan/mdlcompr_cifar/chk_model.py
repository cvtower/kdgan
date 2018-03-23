import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))

from keras.layers import Dense
# Keras layers can be called on TensorFlow tensors:
# x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
# x = Dense(128, activation='relu')(x)
# preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation

from keras.layers import Dropout
# x = Dense(128, activation='relu')(img)
# x = Dropout(0.5)(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# preds = Dense(10, activation='softmax')(x)

from keras.layers import InputLayer
from keras.models import Sequential
# this is our modified Keras model
model = Sequential()
model.add(InputLayer(input_tensor=img, input_shape=(None, 784)))
# build the rest of the model as before
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
preds = model.output


labels = tf.placeholder(tf.float32, shape=(None, 10))

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

from tensorflow.examples.tutorials.mnist import input_data
data_dir = '/home/xiaojie/Projects/data/mnist/'
mnist_data = input_data.read_data_sets(data_dir, one_hot=True)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

from keras.metrics import categorical_accuracy as accuracy
acc_value = tf.reduce_mean(accuracy(labels, preds))

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# Run training loop
with sess.as_default():
  for i in range(10000):
    batch = mnist_data.train.next_batch(50)
    feed_dict = {
      img: batch[0], 
      labels: batch[1],
      K.learning_phase(): 1,
    }
    train_step.run(feed_dict=feed_dict)

    if (i % 100) != 0:
      continue
    feed_dict = {
      img: mnist_data.test.images,
      labels: mnist_data.test.labels,
      K.learning_phase(): 0,
    }
    acc = acc_value.eval(feed_dict=feed_dict)
    print('#%d acc=%.4f' % (i, acc))




