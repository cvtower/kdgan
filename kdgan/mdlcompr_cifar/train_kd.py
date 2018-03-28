from flags import flags
from std_model import STD
from tch_model import TCH

import tensorflow as tf

tn_std = STD(flags, is_training=True)
tn_tch = TCH(flags, is_training=True)
scope = tf.get_variable_scope()
scope.reuse_variables()
vd_std = STD(flags, is_training=False)
vd_tch = TCH(flags, is_training=False)

# tot_params = 0
# for var in tf.trainable_variables():
#   num_params = 1
#   for dim in var.shape:
#     num_params *= dim.value
#   print('%-64s (%d params)' % (var.name, num_params))
#   tot_params += num_params
# print('%-64s (%d params)' % (' '.join(['kd', flags.kd_model]), tot_params))

def main(_):
  tf.logging.info('train kd')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

