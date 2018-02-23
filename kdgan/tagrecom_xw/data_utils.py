from kdgan import config
from kdgan import utils

class YFCC100M(object):
  def __init__(self):
tn_data_sources = utils.get_data_sources(flags, is_training=True, single_source=False)
# tn_data_sources = utils.get_data_sources(flags, is_training=False, single_source=False)
print('#tfrecord=%d for training' % (len(tn_data_sources)))
tn_ts_list = utils.decode_tfrecord(flags, tn_data_sources, shuffle=True)
tn_bt_list = utils.generate_batch(tn_ts_list, flags.batch_size)
tn_user_bt, tn_image_bt, tn_text_bt, tn_label_bt, _ = tn_bt_list

  def next_batch(self):
    while True:
      yield 100