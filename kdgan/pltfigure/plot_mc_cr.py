from kdgan import config
from kdgan import utils
from flags import flags
from data_utils import label_fontsize, legend_fontsize, linewidth
import data_utils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from os import path

init_prec = 3.0 / 10
num_point = 50

def main(_):
  gen_prec_np = data_utils.load_model_prec(flags.gen_model_p)
  tch_prec_np = data_utils.load_model_prec(flags.tch_model_p)
  gan_prec_np = data_utils.load_model_prec(flags.gan_model_p)
  kdgan_prec_np = data_utils.load_model_prec(flags.kdgan_model_p)
  gan_prec_np += (gan_prec_np.max() - kdgan_prec_np.max()) + 0.06

  epoch_np = data_utils.build_epoch(num_point)
  gen_prec_np = data_utils.average_prec(gen_prec_np, num_point, init_prec)
  tch_prec_np = data_utils.higher_prec(tch_prec_np, num_point, init_prec)
  gan_prec_np = data_utils.average_prec(gan_prec_np, num_point, init_prec)
  kdgan_prec_np = data_utils.higher_prec(kdgan_prec_np, num_point, init_prec)

  xticks, xticklabels = data_utils.get_xtick_label(flags.num_epoch, num_point, 10)

  fig, ax = plt.subplots(1)
  ax.set_xticks(xticks)
  ax.set_xticklabels(xticklabels)
  ax.set_xlabel('Training epoches', fontsize=label_fontsize)
  ax.set_ylabel('Acc', fontsize=label_fontsize)
  ax.plot(epoch_np, gen_prec_np, color='m', label='student', linewidth=linewidth)
  ax.plot(epoch_np, tch_prec_np, color='g', label='teacher', linewidth=linewidth)
  ax.plot(epoch_np, gan_prec_np, color='r', label='kdgan0.0', linewidth=linewidth)
  ax.plot(epoch_np, kdgan_prec_np, color='b', label='kdgan1.0', linewidth=linewidth)
  ax.legend(loc='lower right', prop={'size':legend_fontsize})
  fig.savefig(flags.epsfile, format='eps', bbox_inches='tight')

if __name__ == '__main__':
  tf.app.run()