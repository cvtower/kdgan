import numpy as np
import pickle

init_prec = 1.0 / 100

def load_model_prec(model_p):
  prec_list = pickle.load(open(model_p, 'rb'))
  prec_np = np.asarray(prec_list)
  return prec_np

def smooth_prec(prec_np, num_epoch):
  num_batch = prec_np.shape[0]
  epk_batch = num_batch // num_epoch
  prec_list = [init_prec]
  for i in range(num_epoch):
    start = i * epk_batch
    end = (i + 1) * epk_batch
    prec = prec_np[start:end].mean()
    prec_list.append(prec)
  prec_np = np.asarray(prec_list)
  return prec_np

def build_epoch(num_epoch):
  epoch_np = np.arange(1 + num_epoch)
  return epoch_np