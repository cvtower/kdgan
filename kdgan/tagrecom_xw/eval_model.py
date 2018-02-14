from kdgan import config
from kdgan import metric
from kdgan import utils
from gen_model import GEN
import flags

import os
import time

import numpy as np
import tensorflow as tf

from os import path
from tensorflow.contrib import slim

tf.app.flags.DEFINE_string('gen_checkpoint_dir', None, '')
tf.app.flags.DEFINE_string('gen_model_run', None, '')
flags = tf.app.flags.FLAGS

tn_gen = GEN(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, is_training=False)

image_np, label_np, imgid_np = utils.get_valid_data(flags)

gen_model_ckpt = flags.gen_model_ckpt
if flags.gen_checkpoint_dir != None:
  gen_model_ckpt = utils.get_latest_ckpt(flags.gen_checkpoint_dir)

def main(_):
  utils.create_pardir(flags.gen_model_run)
  id_to_label = utils.load_id_to_label(flags.dataset)
  fout = open(flags.gen_model_run, 'w')
  with tf.train.MonitoredTrainingSession() as sess:
    tn_gen.saver.restore(sess, gen_model_ckpt)
    feed_dict = {vd_gen.image_ph:image_np}
    logit_np = sess.run(vd_gen.logits, feed_dict=feed_dict)
    for imgid, logit_np in zip(imgid_np, logit_np):
      sorted_labels = (-logit_np).argsort()
      fout.write('%s' % (imgid))
      for label in sorted_labels:
        fout.write(' %s %.4f' % (id_to_label[label], logit_np[label]))
      fout.write('\n')
  fout.close()

if __name__ == '__main__':
  tf.app.run()