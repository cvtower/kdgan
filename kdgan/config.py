import sys

from os import path

home_dir = path.expanduser('~')
proj_dir = path.join(home_dir, 'Projects')
root_dir = path.join(proj_dir, 'kdgan')
data_dir = path.join(proj_dir, 'data')
yfcc_dir = path.join(data_dir, 'yfcc100m')
temp_dir = path.join(root_dir, 'temp')
surv_dir = path.join(yfcc_dir, 'survey_data')

slim_dir = path.join(root_dir, 'slim')
sys.path.insert(0, slim_dir)

dataset = 'yfcc10k'
image_dir = path.join(yfcc_dir, 'images')
sample_file = path.join(yfcc_dir, 'sample_09')

yfcc10k_dir = path.join(yfcc_dir, 'yfcc10k')
raw_file = path.join(yfcc10k_dir, '%s.raw' % dataset)
data_file = path.join(yfcc10k_dir, '%s.data' % dataset)
train_file = path.join(yfcc10k_dir, '%s.train' % dataset)
valid_file = path.join(yfcc10k_dir, '%s.valid' % dataset)
label_file = path.join(yfcc10k_dir, '%s.label' % dataset)
vocab_file = path.join(yfcc10k_dir, '%s.vocab' % dataset)
image_data_dir = path.join(yfcc10k_dir, 'ImageData')

train_tfrecord = '%s.tfrecord' % train_file
valid_tfrecord = '%s.tfrecord' % valid_file

user_key = 'user'
text_key = 'text'
label_key = 'label'
image_encoded_key = 'image/encoded'
image_format_key = 'image/format'
image_height_key = 'image/height'
image_width_key = 'image/width'
image_file_key = 'image/file'
