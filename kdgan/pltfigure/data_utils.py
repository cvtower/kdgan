import numpy as np
import pickle

label_size = 17
legend_size = 17
tick_size = 15
line_width = 2
marker_size = 8
broken_length = 0.015

def load_model_prec(model_p):
  prec_list = pickle.load(open(model_p, 'rb'))
  prec_np = np.asarray(prec_list)
  return prec_np

def average_prec(prec_np, num_epoch, init_prec):
  num_batch = prec_np.shape[0]
  epk_batch = num_batch // num_epoch
  prec_list = [init_prec]
  for i in range(num_epoch):
    start = i * epk_batch
    end = (i + 1) * epk_batch
    if start > num_batch // 2:
      prec = prec_np[start:end].max()
      middle = int((start + end) / 2)
      prec = prec_np[middle]
    else:
      prec = prec_np[start:end].mean()
    prec_list.append(prec)
  prec_np = np.asarray(prec_list)
  return prec_np

def higher_prec(prec_np, num_epoch, init_prec):
  num_batch = prec_np.shape[0]
  epk_batch = num_batch // num_epoch
  prec_list = [init_prec]
  for i in range(num_epoch):
    start = i * epk_batch
    end = (i + 1) * epk_batch
    prec = prec_np[start:end].max()
    prec_list.append(prec)
  prec_np = np.asarray(prec_list)
  return prec_np

def build_epoch(num_epoch):
  epoch_np = np.arange(1 + num_epoch)
  return epoch_np

def get_xtick_label(num_epoch, num_point, interval):
  xticks, xticklabels = [], []
  for xtick in range(0, num_point + interval, interval):
    xticks.append(xtick)
    xticklabel = str(int(xtick * num_epoch / num_point))
    xticklabels.append(xticklabel)
  return xticks, xticklabels
