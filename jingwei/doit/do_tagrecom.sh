export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export KDGAN_DIR=$BASEDIR/kdgan_xw
export SURVEY_CODE=$KDGAN_DIR/jingwei
export SURVEY_DB=$BASEDIR/kdgan_xw/results/pkls
export MATLAB_PATH=/usr/local
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

export rootpath=$SURVEY_DATA
export codepath=$SURVEY_CODE


./do_knntagrel.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

# ./do_knntagrel.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

# ./do_tagvote.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

# ./do_tagprop.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

# ./do_tagfeat.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu