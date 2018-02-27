from kdgan import config
from kdgan import utils

import pickle
import matplotlib.pyplot as plt
import numpy as np
from os import path

def test():
  gan_filepath = 'gan_vgg_16.csv'
  kdgan_filepath = 'kdgan_vgg_16.csv'
  gan_data = read_data(gan_filepath)
  kdgan_data = read_data(kdgan_filepath)
  epoches, gan_hit, kdgan_hit = [], [], []
  for epoch, (gan, kdgan) in enumerate(zip(gan_data, kdgan_data)):
    if epoch > 90:
      break
    epoches.append(epoch)
    gan_hit.append(gan)
    kdgan_hit.append(kdgan)
  epoches = np.asarray(epoches)
  gan_hit = np.asarray(gan_hit)
  kdgan_hit = np.asarray(kdgan_hit)

  fig, ax = plt.subplots(1)
  ax.set_ylabel('hit@3')
  ax.set_xlabel('epoch')

  ax.plot(epoches, gan_hit, color='r', label=r'gan')
  ax.plot(epoches, kdgan_hit, color='b', label=r'kdgan')
  ax.legend(loc='lower right')
  
  fig.savefig('gan_kdgan.eps', format='eps', bbox_inches='tight')

  filepath = path.join(config.logs_dir, 'mdlcompr_check_kd.txt')
  fin = open(filepath)
  line = fin.readline()
  fields = line.split()
  sl_label, kd_label = fields[0], fields[1]
  batches, sl_data, kd_data = [], [], []
  for line in fin.readlines():
    fields = line.split()
    batches.append(int(fields[0]))
    sl_data.append(float(fields[1]))
    kd_data.append(float(fields[2]))
  fin.close()
  start = 49999
  batches = np.asarray(batches)[start:]
  sl_data = np.asarray(sl_data)[start:]
  kd_data = np.asarray(kd_data)[start:]

  fig, ax = plt.subplots(1)
  ax.set_ylabel('accuracy')
  ax.set_xlabel('#batch')
  ax.plot(batches, sl_data, color='r', label=sl_label)
  ax.plot(batches, kd_data, color='b', label=kd_label)
  ax.legend(loc='lower right')
  fig.savefig('sl_kd.eps', format='eps', bbox_inches='tight')

def main():
  name = 'tagrecom_yfcc10k_tch@2'
  tagrecom_yfcc10k_tch_p = path.join(config.pickle_dir, '%s.p' % name)
  prec_list = pickle.load(open(tagrecom_yfcc10k_tch_p, 'rb'))
  batches = list(range(1, 1 + len(prec_list)))

  fig, ax = plt.subplots(1)
  ax.plot(batches, prec_list, color='r', label=name)
  ax.legend(loc='lower right')

  tagrecom_yfcc10k_tch_eps = path.join(config.picture_dir, '%s.eps' % name)
  utils.create_pardir(tagrecom_yfcc10k_tch_eps)
  fig.savefig(tagrecom_yfcc10k_tch_eps, format='eps', bbox_inches='tight')

if __name__ == '__main__':
  main()