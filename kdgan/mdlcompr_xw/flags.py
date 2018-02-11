import tensorflow as tf

# dataset
tf.app.flags.DEFINE_integer('channels', 1, '')
tf.app.flags.DEFINE_integer('image_size', 28, '')
tf.app.flags.DEFINE_integer('num_label', 10, '')
tf.app.flags.DEFINE_integer('train_size', 60000, '')
tf.app.flags.DEFINE_integer('valid_size', 0, '')
tf.app.flags.DEFINE_string('dataset_dir', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
# image model
tf.app.flags.DEFINE_string('dis_model_name', None, '')
tf.app.flags.DEFINE_string('gen_model_name', None, '')
tf.app.flags.DEFINE_string('tch_model_name', None, '')
tf.app.flags.DEFINE_float('dis_keep_prob', 0.50, '')
tf.app.flags.DEFINE_float('gen_keep_prob', 0.88, '')
tf.app.flags.DEFINE_float('tch_keep_prob', 0.50, '')
tf.app.flags.DEFINE_float('dis_weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('gen_weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_float('tch_weight_decay', 0.00004, 'l2 coefficient')
tf.app.flags.DEFINE_string('dis_ckpt_file', None, '')
tf.app.flags.DEFINE_string('gen_ckpt_file', None, '')
tf.app.flags.DEFINE_string('tch_ckpt_file', None, '')
# optimization
tf.app.flags.DEFINE_integer('num_epoch', 200, '')
tf.app.flags.DEFINE_integer('batch_size', 128, '')
tf.app.flags.DEFINE_string('optimizer', 'adam', 'adam|rmsprop|sgd')
tf.app.flags.DEFINE_float('dis_learning_rate', 1e-3, '')
tf.app.flags.DEFINE_float('gen_learning_rate', 1e-3, '')
tf.app.flags.DEFINE_float('tch_learning_rate', 1e-3, '')
# kd or gan
tf.app.flags.DEFINE_float('kd_hard_pct', 0.7, '')
tf.app.flags.DEFINE_float('kd_soft_pct', 0.3, '')
tf.app.flags.DEFINE_float('temperature', 3.0, '')
tf.app.flags.DEFINE_float('noisy_ratio', 0.005, '')
tf.app.flags.DEFINE_float('noisy_sigma', 0.005, '')
tf.app.flags.DEFINE_string('kd_model', 'mimic', 'mimic|distn|noisy')

flags = tf.app.flags.FLAGS
