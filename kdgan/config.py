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
init_sample_filepath = path.join(yfcc_dir, 'sample_00')
sample_filepath = path.join(yfcc_dir, 'sample_09')

yfcc10k_dir = path.join(yfcc_dir, 'yfcc10k')
raw_filepath = path.join(yfcc10k_dir, '%s.raw' % dataset)
data_filepath = path.join(yfcc10k_dir, '%s.data' % dataset)
train_filepath = path.join(yfcc10k_dir, '%s.train' % dataset)
valid_filepath = path.join(yfcc10k_dir, '%s.valid' % dataset)
label_filepath = path.join(yfcc10k_dir, '%s.label' % dataset)
vocab_filepath = path.join(yfcc10k_dir, '%s.vocab' % dataset)



