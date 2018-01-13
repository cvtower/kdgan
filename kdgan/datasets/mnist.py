from kdgan import config

import tensorflow as tf
from datasets import download_and_convert_mnist

tf.app.flags.DEFINE_string('dataset_dir', None, '')
flags = tf.app.flags.FLAGS

def main(_):
  if not flags.dataset_dir:
    raise ValueError('no --dataset_dir')
  download_and_convert_mnist.run(flags.dataset_dir)

if __name__ == '__main__':
  tf.app.run()
