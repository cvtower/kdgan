from kdgan import config
from kdgan import utils
from flags import flags
from data_utils import CIFAR_TF, KERAS_DG

from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
import keras
import math
import os
import time

cifar_tf = CIFAR_TF(flags)
keras_dg = KERAS_DG(flags)

tn_num_batch = int(flags.num_epoch * flags.train_size / flags.batch_size)
print('#tn_batch=%d' % (tn_num_batch))
eval_interval = int(math.ceil(flags.train_size / flags.batch_size))

# start_time = time.time()
# for tn_batch in range(tn_num_batch):
#   tn_image_np, tn_label_np = keras_dg.next_batch()
#   if (tn_batch + 1) % eval_interval != 0 and (tn_batch + 1) != tn_num_batch:
#     continue
#   print('#batch=%d' % (tn_batch))
# duration = time.time() - start_time
# print('keras_dg #tn_batch=%d time=%.4fs' % (tn_num_batch, duration))

sess = tf.train.MonitoredTrainingSession()
start_time = time.time()
for tn_batch in range(tn_num_batch):
  tn_image_np, tn_label_np = cifar_tf.next_batch(sess)
  if (tn_batch + 1) % eval_interval != 0 and (tn_batch + 1) != tn_num_batch:
    continue
  print('#batch=%d' % (tn_batch))
duration = time.time() - start_time
print('cifar_tf #tn_batch=%d time=%.4fs' % (tn_num_batch, duration))
sess.close()















