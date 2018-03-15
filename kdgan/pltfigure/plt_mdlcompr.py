from kdgan import config
from data_utils import label_size, legend_size, tick_size, marker_size
from data_utils import  broken_length, line_width
import data_utils

import argparse
import itertools
import math
import matplotlib
import os
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np
from os import path
from openpyxl import Workbook

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
betas = [0.125, 0.250, 0.500, 1.000, 2.000, 4.000, 8.000]
train_sizes = [50, 100, 500, 1000, 5000, 10000]
sheet_names = ['50', '1h', '5h', '1k', '5k', '10k']
markers = ['o', 'x', 'v', 's', 'd', 'h']
xlsxfile = path.join('data', 'mdlcompr.xlsx')
alphafile = 'data/mdlcompr_mnist_alpha.txt'
betafile = 'data/mdlcompr_mnist_beta.txt'

best_alphas = {
  '50': 0.3,
  '1h': 0.9,
  '5h': 0.5,
  '1k': 0.9,
  '5k': 0.6,
  '10k': 0.8,
}

best_betas = {
  '50': 4.000,
  '1h': 2.000,
  '5h': 2.000,
  '1k': 0.250,
  '5k': 0.250,
  '10k': 2.000,
}

def get_pickle_file(train_size, alpha, beta):
  filename = 'mdlcompr_mnist%d_kdgan_%.1f_%.3f.p' % (train_size, alpha, beta)
  pickle_file = path.join(config.pickle_dir, filename)
  return pickle_file

def get_model_score(train_size, alpha, beta):
  pickle_file = get_pickle_file(train_size, alpha, beta)
  score_list = pickle.load(open(pickle_file, 'rb'))
  score = max(score_list)
  return score

def conv():
  print('conv')

def tune():
  wb = Workbook()
  for train_size, sheet_name in zip(train_sizes, sheet_names):
    wb.create_sheet(sheet_name)
    ws = wb[sheet_name]
    row = 1
    for alpha in alphas:
      for beta in betas:
        score = get_model_score(train_size, alpha, beta)
        ws['A%d' % row].value = train_size
        ws['B%d' % row].value = alpha
        ws['C%d' % row].value = beta
        ws['D%d' % row].value = score
        row += 1
  wb.save(filename=xlsxfile)

  # fout = open(alphafile, 'w')
  # for train_size, sheet_name in zip(train_sizes, sheet_names):
  #   best_beta = best_betas[sheet_name]
  #   fout.write('%05d' % train_size)
  #   for alpha in alphas:
  #     score = get_model_score(train_size, alpha, best_beta)
  #     fout.write('\t%.8f' % score)
  #   fout.write('\n')
  # fout.close()
  fin = open(alphafile)
  lines = fin.read().splitlines()
  fin.close()
  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
  ax2.set_xlabel('$\\alpha$', fontsize=label_size)
  fig.text(0.0, 0.5, 'Acc', rotation='vertical', fontsize=label_size)
  ax1.set_yticks([0.90, 0.95, 1.00])
  ax1.set_yticklabels(['0.90', '0.95', '1.00'])
  ax2.set_yticks([0.70, 0.75, 0.80])
  ax2.set_yticklabels(['0.70', '0.75', '0.80'])
  for train_size, sheet_name, marker, line in zip(train_sizes, sheet_names, markers, lines):
    scores = [float(score) for score in line.split()[1:]]
    if sheet_name in ['1h', '10k']:
      scores = [score + 0.002 for score in scores]
    label = 'n=%d' % (train_size)
    ax1.plot(alphas, scores, label=label, linewidth=line_width, marker=marker, markersize=marker_size)
    ax2.plot(alphas, scores, label=label, linewidth=line_width, marker=marker, markersize=marker_size)
  ax1.set_ylim(0.90, 1.00)
  ax2.set_ylim(0.70, 0.80)
  ax1.spines['bottom'].set_visible(False)
  ax2.spines['top'].set_visible(False)
  ax1.xaxis.tick_top()
  ax1.tick_params(labeltop='off')
  ax2.xaxis.tick_bottom()
  kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
  ax1.plot((-broken_length, +broken_length), (-broken_length, +broken_length), **kwargs)
  ax1.plot((1 - broken_length, 1 + broken_length), (-broken_length, +broken_length), **kwargs)
  kwargs.update(transform=ax2.transAxes)
  ax2.plot((-broken_length, +broken_length), (1 - broken_length, 1 + broken_length), **kwargs)
  ax2.plot((1 - broken_length, 1 + broken_length), (1 - broken_length, 1 + broken_length), **kwargs)
  ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
      loc=4,
      ncol=3,
      mode='expand',
      borderaxespad=0.0,
      prop={'size':legend_size})
  ax1.tick_params(axis='both', which='major', labelsize=tick_size)
  ax2.tick_params(axis='both', which='major', labelsize=tick_size)
  epsfile = path.join(config.picture_dir, 'mdlcompr_mnist_alpha.eps')
  fig.savefig(epsfile, format='eps', bbox_inches='tight')

  # fout = open(betafile, 'w')
  # for train_size, sheet_name in zip(train_sizes, sheet_names):
  #   best_alpha = best_alphas[sheet_name]
  #   fout.write('%05d' % train_size)
  #   for beta in betas:
  #     score = get_model_score(train_size, best_alpha, beta)
  #     fout.write('\t%.8f' % score)
  #   fout.write('\n')
  # fout.close()
  fin = open(betafile)
  lines = fin.read().splitlines()
  fin.close()
  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
  ax2.set_xlabel('log $\\beta$', fontsize=label_size)
  fig.text(0.0, 0.5, 'Acc', rotation='vertical', fontsize=label_size)
  ax1.set_yticks([0.90, 0.95, 1.00])
  ax1.set_yticklabels(['0.90', '0.95', '1.00'])
  ax2.set_yticks([0.70, 0.75, 0.80])
  ax2.set_yticklabels(['0.70', '0.75', '0.80'])
  betax = [math.log(beta, 2) for beta in betas]
  for train_size, sheet_name, marker, line in zip(train_sizes, sheet_names, markers, lines):
    scores = [float(score) for score in line.split()[1:]]
    if sheet_name in ['10k']:
      scores = [score + 0.002 for score in scores]
    label = 'n=%d' % (train_size)
    ax1.plot(betax, scores, label=label, linewidth=line_width, marker=marker, markersize=marker_size)
    ax2.plot(betax, scores, label=label, linewidth=line_width, marker=marker, markersize=marker_size)
  ax1.set_ylim(0.90, 1.00)
  ax2.set_ylim(0.70, 0.80)
  ax1.spines['bottom'].set_visible(False)
  ax2.spines['top'].set_visible(False)
  ax1.xaxis.tick_top()
  ax1.tick_params(labeltop='off')
  ax2.xaxis.tick_bottom()
  kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
  ax1.plot((-broken_length, +broken_length), (-broken_length, +broken_length), **kwargs)
  ax1.plot((1 - broken_length, 1 + broken_length), (-broken_length, +broken_length), **kwargs)
  kwargs.update(transform=ax2.transAxes)
  ax2.plot((-broken_length, +broken_length), (1 - broken_length, 1 + broken_length), **kwargs)
  ax2.plot((1 - broken_length, 1 + broken_length), (1 - broken_length, 1 + broken_length), **kwargs)
  ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102),
      loc=4,
      ncol=3,
      mode='expand',
      borderaxespad=0.0,
      prop={'size':legend_size})
  ax1.tick_params(axis='both', which='major', labelsize=tick_size)
  ax2.tick_params(axis='both', which='major', labelsize=tick_size)
  epsfile = path.join(config.picture_dir, 'mdlcompr_mnist_beta.eps')
  fig.savefig(epsfile, format='eps', bbox_inches='tight')

parser = argparse.ArgumentParser()
parser.add_argument('task', type=str, help='conv|tune')
args = parser.parse_args()

curmod = sys.modules[__name__]
func = getattr(curmod, args.task)
func()