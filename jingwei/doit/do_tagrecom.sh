export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan_xw/jingwei
export SURVEY_DB=$BASEDIR/kdgan_xw/results/runs
export MATLAB_PATH=/usr/local
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

export rootpath=$SURVEY_DATA
export codepath=$SURVEY_CODE

# ./do_knntagrel.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

# ./do_tagvote.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

./do_tagprop.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

# ./do_tagfeat.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu