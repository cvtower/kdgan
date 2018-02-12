from kdgan import config
from kdgan import metric
from kdgan import utils
from flags import flags
from gen_model import GEN

from os import path
from tensorflow.contrib import slim
import os
import time
import numpy as np
import tensorflow as tf

tn_size = utils.get_tn_size(flags.dataset)
tn_num_batch = int(flags.num_epoch * tn_size / flags.batch_size)
eval_interval = int(tn_size / flags.batch_size)
print('#batch=%d #interval=%d' % (tn_num_batch, eval_interval))
# exit()

precomputed_dir = utils.get_precomputed_dir(flags.dataset)
filename_tmpl = 'yfcc10k_%s.valid.%s.npy'
vd_image_file = filename_tmpl % (flags.image_model, 'image')
vd_image_np = np.load(path.join(precomputed_dir, vd_image_file))
vd_label_file = filename_tmpl % (flags.image_model, 'label')
vd_label_np = np.load(path.join(precomputed_dir, vd_label_file))
vd_imgid_file = filename_tmpl % (flags.image_model, 'imgid')
vd_imgid_np = np.load(path.join(precomputed_dir, vd_imgid_file))
# print(vd_image_np.shape, vd_label_np.shape, vd_imgid_np.shape)
# exit()

tn_gen = GEN(flags, is_training=True)
tf.summary.scalar(tn_gen.learning_rate.name, tn_gen.learning_rate)
tf.summary.scalar(tn_gen.pre_loss.name, tn_gen.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

scope = tf.get_variable_scope()
scope.reuse_variables()
vd_gen = GEN(flags, is_training=False)

for variable in tf.trainable_variables():
  num_params = 1
  for dim in variable.shape:
    num_params *= dim.value
  print('%-50s (%d params)' % (variable.name, num_params))
# exit()

data_sources = utils.get_data_sources(flags, is_training=True)
print('tn: #tfrecord=%d' % (len(data_sources)))
ts_list = utils.decode_tfrecord(flags, data_sources, shuffle=True)
bt_list = utils.generate_batch(ts_list, flags.batch_size)
user_bt, image_bt, text_bt, label_bt, file_bt = bt_list

def main(_):
  best_hit = 0.0
  writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    start = time.time()
    for batch_t in range(tn_num_batch):
      image_np_t, label_np_t = sess.run([image_bt, label_bt])
      feed_dict = {tn_gen.image_ph:image_np_t, tn_gen.hard_label_ph:label_np_t}
      _, summary = sess.run([tn_gen.pre_update, summary_op], feed_dict=feed_dict)
      writer.add_summary(summary, batch_t)

      batch = batch_t + 1
      remain = (batch * flags.batch_size) % tn_size
      epoch = (batch * flags.batch_size) // tn_size
      if remain == 0:
        pass
        # print('%d\t%d\t%d' % (epoch, batch, remain))
      elif (tn_size - remain) < flags.batch_size:
        epoch = epoch + 1
        # print('%d\t%d\t%d' % (epoch, batch, remain))
      else:
        continue
      # if (batch_t + 1) % eval_interval != 0:
      #     continue

      feed_dict = {vd_gen.image_ph:vd_image_np}
      logit_np_v, = sess.run([vd_gen.logits], feed_dict=feed_dict)
      # print(logit_np_v.shape, vd_label_np.shape)
      hit_v = metric.compute_hit(logit_np_v, vd_label_np, flags.cutoff)

      if hit_v < best_hit:
        continue
      tot_time = time.time() - start
      best_hit = hit_v
      print('#%03d curbst=%.4f time=%.0fs' % (epoch, hit_v, tot_time))
      tn_gen.saver.save(utils.get_session(sess), flags.gen_model_ckpt)
  print('bsthit=%.4f' % (best_hit))

  id_to_label = utils.load_id_to_label(flags.dataset)
  # print(id_to_label)
  fout = open(flags.gen_model_eval, 'w')
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    tn_gen.saver.restore(sess, flags.gen_model_ckpt)
    feed_dict = {vd_gen.image_ph:vd_image_np}
    logit_np_v, = sess.run([vd_gen.logits], feed_dict=feed_dict)
    # print(logit_np_v.shape, vd_label_np.shape)
    hit_v = metric.compute_hit(logit_np_v, vd_label_np, flags.cutoff)
    for imgid, logit_np in zip(vd_imgid_np, logit_np_v):
      sorted_labels = (-logit_np).argsort()
      fout.write('%s' % (imgid))
      for label in sorted_labels:
        fout.write(' %s %.4f' % (id_to_label[label], logit_np[label]))
      fout.write('\n')
  fout.close()

if __name__ == '__main__':
  tf.app.run()