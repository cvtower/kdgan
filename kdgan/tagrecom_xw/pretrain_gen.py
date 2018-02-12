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
vd_image_file = filename_tmpl % (flags.gen_model_name, 'image')
vd_image_np = np.load(path.join(precomputed_dir, vd_image_file))
vd_label_file = filename_tmpl % (flags.gen_model_name, 'label')
vd_label_np = np.load(path.join(precomputed_dir, vd_label_file))
vd_imgid_file = filename_tmpl % (flags.gen_model_name, 'imgid')
vd_imgid_np = np.load(path.join(precomputed_dir, vd_imgid_file))
print(vd_image_np.shape, vd_label_np.shape, vd_imgid_np.shape)
exit()

gen_t = GEN(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
gen_v = GEN(flags, is_training=False)

tf.summary.scalar(gen_t.learning_rate.name, gen_t.learning_rate)
tf.summary.scalar(gen_t.pre_loss.name, gen_t.pre_loss)
summary_op = tf.summary.merge_all()
init_op = tf.global_variables_initializer()

def train():
  for variable in tf.trainable_variables():
    num_params = 1
    for dim in variable.shape:
      num_params *= dim.value
    print('%-50s (%d params)' % (variable.name, num_params))

  data_sources_t = utils.get_data_sources(flags, is_training=True)
  # data_sources_v = utils.get_data_sources(flags, is_training=False)
  # print('tn: #tfrecord=%d\nvd: #tfrecord=%d' % (len(data_sources_t), len(data_sources_v)))
  
  ts_list_t = utils.decode_tfrecord(flags, data_sources_t, shuffle=True)
  # ts_list_v = utils.decode_tfrecord(flags, data_sources_v, shuffle=False)
  bt_list_t = utils.generate_batch(ts_list_t, flags.batch_size)
  # bt_list_v = utils.generate_batch(ts_list_v, config.valid_batch_size)
  user_bt_t, image_bt_t, text_bt_t, label_bt_t, file_bt_t = bt_list_t
  # user_bt_v, image_bt_v, text_bt_v, label_bt_v, file_bt_v = bt_list_v

  best_hit_v = -np.inf
  start = time.time()
  with tf.Session() as sess:
    sess.run(init_op)
    writer = tf.summary.FileWriter(config.logs_dir, graph=tf.get_default_graph())
    with slim.queues.QueueRunners(sess):
      for batch_t in range(tn_num_batch):
        image_np_t, label_np_t = sess.run([image_bt_t, label_bt_t])
        feed_dict = {gen_t.image_ph:image_np_t, gen_t.hard_label_ph:label_np_t}
        _, summary = sess.run([gen_t.pre_update, summary_op], feed_dict=feed_dict)
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

        # hit_v = []
        # for batch_v in range(num_batch_v):
        #   vd_image_np, vd_label_np = sess.run([image_bt_v, label_bt_v])
        #   feed_dict = {gen_v.image_ph:vd_image_np}
        #   logit_np_v, = sess.run([gen_v.logits], feed_dict=feed_dict)
        #   hit_bt = metric.compute_hit(logit_np_v, vd_label_np, flags.cutoff)
        #   hit_v.append(hit_bt)
        # hit_v = np.mean(hit_v)

        feed_dict = {gen_v.image_ph:vd_image_np}
        logit_np_v, = sess.run([gen_v.logits], feed_dict=feed_dict)
        # print(logit_np_v.shape, vd_label_np.shape)
        hit_v = metric.compute_hit(logit_np_v, vd_label_np, flags.cutoff)

        if hit_v < best_hit_v:
          continue
        tot_time = time.time() - start
        best_hit_v = hit_v
        print('#%03d curbst=%.4f time=%.0fs' % (epoch, hit_v, tot_time))
        gen_t.saver.save(sess, flags.gen_model_ckpt)
  print('bsthit=%.4f' % (best_hit_v))

def test():
  id_to_label = utils.load_id_to_label(flags.dataset)
  # print(id_to_label)
  fout = open(flags.gen_model_eval, 'w')
  with tf.train.MonitoredTrainingSession() as sess:
    sess.run(init_op)
    gen_t.saver.restore(sess, flags.gen_model_ckpt)
    feed_dict = {gen_v.image_ph:vd_image_np}
    logit_np_v, = sess.run([gen_v.logits], feed_dict=feed_dict)
    # print(logit_np_v.shape, vd_label_np.shape)
    hit_v = metric.compute_hit(logit_np_v, vd_label_np, flags.cutoff)
    for imgid, logit_np in zip(vd_imgid_np, logit_np_v):
      sorted_labels = (-logit_np).argsort()
      fout.write('%s' % (imgid))
      for label in sorted_labels:
        fout.write(' %s %.4f' % (id_to_label[label], logit_np[label]))
      fout.write('\n')
  fout.close()

def main(_):
  if flags.task == 'train':
    train()
  else:
    test()

if __name__ == '__main__':
  tf.app.run()