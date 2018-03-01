from kdgan import config
from kdgan import utils
from flags import flags
from data_utils import label_fontsize, legend_fontsize, linewidth
import data_utils

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import tensorflow as tf
from os import path
from openpyxl import Workbook

init_prec = 3.0 / 10
num_point = 50

def get_highest_prec(train_size, privilege_weight, distilled_weight):
  template = 'mdlcompr_mnist%d_kdgan_%.1f_%.1f.p'
  filename = template % (train_size, privilege_weight, distilled_weight)
  filepath = path.join(config.pickle_dir, filename)
  prec_np = data_utils.load_model_prec(filepath)
  prec = prec_np.max()
  return prec

def main(_):
  train_sizes = [5e1, 1e2, 5e2, 1e3, 5e3, 1e4]
  train_sizes = [5e1, 1e2, 5e2, 1e3]
  sheet_names = ['50', '1h', '5h', '1k', '5k', '10k']
  privilege_weights = [0.1, 0.3, 0.5, 0.7, 0.9]
  distilled_weights = [0.1, 0.5, 1.0, 5.0, 10.]
  len_privilege = len(privilege_weights)
  len_distilled = len(distilled_weights)

  fig = plt.figure()
  ax = fig.gca(projection='3d')
  xlsxfile = 'mdlcompr_pe.xlsx'
  wb = Workbook()
  for train_size, sheet_name in zip(train_sizes, sheet_names):
    wb.create_sheet(sheet_name)
    ws = wb[sheet_name]
    row = 1

    x, y = np.meshgrid(privilege_weights, distilled_weights)
    y = np.log10(y)
    z = np.zeros((len_distilled, len_privilege))
    i = 0
    for privilege_weight in privilege_weights:
      j = 0
      for distilled_weight in distilled_weights:
        prec = get_highest_prec(train_size, privilege_weight, distilled_weight)
        ws['A%d' % row].value = train_size
        ws['B%d' % row].value = privilege_weight
        ws['C%d' % row].value = distilled_weight
        ws['D%d' % row].value = prec
        row += 1
        z[i, j] = prec
        j += 1
      i += 1
    ax.plot_surface(x, y, z)
  wb.save(filename=xlsxfile)

  plt.show()

if __name__ == '__main__':
  tf.app.run()