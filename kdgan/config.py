import sys
from os import path

config_file = path.realpath(__file__)
kdgan_dir = path.dirname(config_file)
pypkg_dir = path.dirname(kdgan_dir)
print('config pypkg:%s' % (pypkg_dir))

home_dir = path.expanduser('~')
proj_dir = path.join(home_dir, 'Projects')
data_dir = path.join(proj_dir, 'data')
yfcc_dir = path.join(data_dir, 'yfcc100m')
# pypkg_dir = path.join(proj_dir, 'kdgan')
surv_dir = path.join(yfcc_dir, 'survey_data')
# kdgan_dir = path.join(pypkg_dir, 'kdgan')

logs_dir = path.join(kdgan_dir, 'logs')
temp_dir = path.join(kdgan_dir, 'temp')
ckpt_dir = path.join(kdgan_dir, 'checkpoints')
image_dir = path.join(yfcc_dir, 'images')
mnist_dir = path.join(data_dir, 'mnist')

slim_dir = path.join(pypkg_dir, 'slim')
sys.path.insert(0, slim_dir)

rawtag_file = path.join(yfcc_dir, 'sample_00')
sample_file = path.join(yfcc_dir, 'sample_09')
tfrecord_tmpl = '{0}_{1}_{2:03d}.{3}.tfrecord'

user_key = 'user'
image_key = 'image'
text_key = 'text'
label_key = 'label'
file_key = 'file'
unk_token = 'unk'
pad_token = ' '

# channels = 3
# num_label = 100
num_readers = 4
num_threads = 4
num_preprocessing_threads = 4

# train_batch_size = 32
valid_batch_size = 100








