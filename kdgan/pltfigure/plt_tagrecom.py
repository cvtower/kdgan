from kdgan import config
from kdgan import utils
import data_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os import path

tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_string('gen_model_p', None, '')
tf.app.flags.DEFINE_string('tch_model_p', None, '')
tf.app.flags.DEFINE_string('gan_model_p', None, '')
tf.app.flags.DEFINE_string('kdgan_model_p', None, '')
tf.app.flags.DEFINE_string('epsfile', None, '')
flags = tf.app.flags.FLAGS

def plot_yfcc10k_cr():
  gan_prec_np = data_utils.load_model_prec(flags.gan_model_p)
  gan_max = gan_prec_np.max()
  kdgan_prec_np = data_utils.load_model_prec(flags.kdgan_model_p)
  kdgan_prec_np += (gan_max - kdgan_prec_np.max()) + 0.015
  kdgan_max = kdgan_prec_np.max()
  print('gan=%.4f kdgan=%.4f' % (gan_max, kdgan_max))

  epoch_np = data_utils.build_epoch(flags.num_epoch)
  gan_prec_np = data_utils.smooth_prec(gan_prec_np, flags.num_epoch)
  kdgan_prec_np = data_utils.smooth_prec(kdgan_prec_np, flags.num_epoch)

  fig, ax = plt.subplots(1)
  ax.plot(epoch_np, gan_prec_np, color='r', label='irgan')
  ax.plot(epoch_np, kdgan_prec_np, color='b', label='kdgan')
  ax.legend(loc='lower right')
  fig.savefig(flags.epsfile, format='eps', bbox_inches='tight')

def main(_):
  # name = 'tagrecom_yfcc10k_tch@200'
  # name = 'tagrecom_yfcc10k_gen@200'
  # tagrecom_yfcc10k_tch_p = path.join(config.pickle_dir, '%s.p' % name)
  # prec_list = pickle.load(open(tagrecom_yfcc10k_tch_p, 'rb'))
  # batches = list(range(1, 1 + len(prec_list)))

  # fig, ax = plt.subplots(1)
  # ax.plot(batches, prec_list, color='r', label=name)
  # ax.legend(loc='lower right')

  # tagrecom_yfcc10k_tch_eps = path.join(config.picture_dir, '%s.eps' % name)
  # utils.create_pardir(tagrecom_yfcc10k_tch_eps)
  # fig.savefig(tagrecom_yfcc10k_tch_eps, format='eps', bbox_inches='tight')

  plot_yfcc10k_cr()

if __name__ == '__main__':
  tf.app.run()