import sys

from os import path

home_dir = path.expanduser('~')
proj_dir = path.join(home_dir, 'Projects')
data_dir = path.join(proj_dir, 'data')
yfcc_dir = path.join(data_dir, 'yfcc100m')
root_dir = path.join(proj_dir, 'kdgan')
surv_dir = path.join(yfcc_dir, 'survey_data')

slim_dir = path.join(root_dir, 'slim')
sys.path.insert(0, slim_dir)

kdgan_dir = path.join(root_dir, 'kdgan')
logs_dir = path.join(kdgan_dir, 'logs')
temp_dir = path.join(kdgan_dir, 'temp')
ckpt_dir = path.join(kdgan_dir, 'checkpoints')

image_dir = path.join(yfcc_dir, 'images')
rawtag_file = path.join(yfcc_dir, 'sample_00')
sample_file = path.join(yfcc_dir, 'sample_09')

# dataset = 'yfcc10k'
dataset = 'yfcc20k'

dataset_dir = path.join(yfcc_dir, dataset)
raw_file = path.join(dataset_dir, '%s.raw' % dataset)
data_file = path.join(dataset_dir, '%s.data' % dataset)
train_file = path.join(dataset_dir, '%s.train' % dataset)
valid_file = path.join(dataset_dir, '%s.valid' % dataset)
label_file = path.join(dataset_dir, '%s.label' % dataset)
vocab_file = path.join(dataset_dir, '%s.vocab' % dataset)
image_data_dir = path.join(dataset_dir, 'ImageData')

train_tfrecord = '%s.tfrecord' % train_file
valid_tfrecord = '%s.tfrecord' % valid_file

user_key = 'user'
image_key = 'image'
text_key = 'text'
label_key = 'label'
file_key = 'file'

unk_token = 'unk'
pad_token = ' '
num_threads = 4
channels = 3

num_label = 100
vocab_size = 7281
train_data_size = 8000
valid_data_size = 2000
train_batch_size = 50
valid_batch_size = 500

precomputed_dir = path.join(dataset_dir, 'Precomputed')
tfrecord_tmpl = '{0}_{1}_{2:03d}.{3}.tfrecord'








