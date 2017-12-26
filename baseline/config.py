import os
import sys

from os import path

home_dir = path.expanduser('~')
root_dir = path.join(home_dir, 'Projects/kdgan')
data_dir = path.join(root_dir, 'data')
logs_dir = path.join(root_dir, 'logs')


name = 'train10k'
survey_code = path.join(root_dir, 'jingwei')
survey_data = path.join(data_dir, name)
survey_db = path.join(data_dir, name)
matlab_path = '/usr/local/bin'
if not path.isfile(path.join(matlab_path, 'matlab')):
    matlab_path = '/Applications/MATLAB_R2017b.app/bin'

sys.path.insert(0, survey_code)

os.environ['SURVEY_CODE'] = survey_code
os.environ['SURVEY_DATA'] = survey_data
os.environ['SURVEY_DB'] = survey_db
os.environ['MATLAB_PATH'] = matlab_path

