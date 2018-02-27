from kdgan import config
from kdgan import utils

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os import path

tf.app.flags.DEFINE_string('gen_model_p', None, '')
tf.app.flags.DEFINE_string('tch_model_p', None, '')
tf.app.flags.DEFINE_string('gan_model_p', None, '')
tf.app.flags.DEFINE_string('kdgan_model_p', None, '')
tf.app.flags.DEFINE_string('epsfile', None, '')
flags = tf.app.flags.FLAGS

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

  # gen_prec_list = pickle.load(open(flags.gen_model_p, 'rb'))
  # gen_batches = list(range(1, 1 + len(gen_prec_list)))
  # tch_prec_list = pickle.load(open(flags.tch_model_p, 'rb'))
  # tch_batches = list(range(1, 1 + len(tch_prec_list)))
  # fig, ax = plt.subplots(1)
  # ax.plot(gen_batches, gen_prec_list, color='r', label='gen')
  # ax.plot(tch_batches, tch_prec_list, color='b', label='tch')
  # ax.legend(loc='lower right')
  # fig.savefig(flags.epsfile, format='eps', bbox_inches='tight')

  

if __name__ == '__main__':
  tf.app.run()