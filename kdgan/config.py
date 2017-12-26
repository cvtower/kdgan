import sys

from os import path

home_dir = path.expanduser('~')
proj_dir = path.join(home_dir, 'Projects')
root_dir = path.join(proj_dir, 'kdgan')
data_dir = path.join(proj_dir, 'data')
yfcc_dir = path.join(data_dir, 'yfcc100m')
temp_dir = path.join(root_dir, 'temp')

slim_dir = path.join(root_dir, 'slim')
sys.path.insert(0, slim_dir)

dataset = 'yfcc10k'
image_dir = path.join(yfcc_dir, 'images')
sample_filepath = path.join(yfcc_dir, 'sample_09')
data_filepath = path.join(yfcc_dir, '%s.data'%dataset)
train_filepath = path.join(yfcc_dir, '%s.train'%dataset)
valid_filepath = path.join(yfcc_dir, '%s.valid'%dataset)
