import os
import sys

from os import path

home_dir = path.expanduser('~')
proj_dir = path.join(home_dir, 'Projects')
root_dir = path.join(proj_dir, 'kdgan')
data_dir = path.join(proj_dir, 'data')
yfcc_dir = path.join(data_dir, 'yfcc100m')
temp_dir = path.join(root_dir, 'temp')
surv_dir = path.join(yfcc_dir, 'survey_data')
logs_dir = path.join(root_dir, 'logs')

survey_code = path.join(root_dir, 'jingwei')
matlab_path = '/usr/local/bin'
# if not path.isfile(path.join(matlab_path, 'matlab')):
#     matlab_path = '/Applications/MATLAB_R2017b.app/bin'

sys.path.insert(0, survey_code)

os.environ['SURVEY_CODE'] = survey_code
os.environ['SURVEY_DATA'] = surv_dir
os.environ['SURVEY_DB'] = logs_dir
os.environ['MATLAB_PATH'] = matlab_path

dataset = 'yfcc10k'
dataset_dir = path.join(yfcc_dir, 'yfcc10k')
data_filepath = path.join(dataset_dir, '%s.data'%dataset)
train_filepath = path.join(dataset_dir, '%s.train'%dataset)
valid_filepath = path.join(dataset_dir, '%s.valid'%dataset)

