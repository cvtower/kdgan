from kdgan import config

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.layers import InputLayer

image_ph = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

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
print(type(labels))
losses = model.losses
print(type(losses))