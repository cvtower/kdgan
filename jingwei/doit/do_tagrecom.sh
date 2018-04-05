export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export KDGAN_DIR=$BASEDIR/kdgan_xw
export SURVEY_CODE=$KDGAN_DIR/jingwei
export SURVEY_DB=$BASEDIR/kdgan_xw/results/temp
export MATLAB_PATH=/usr/local
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

export rootpath=$SURVEY_DATA
export codepath=$SURVEY_CODE


# ./do_knntagrel.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu
# ./do_tagvote.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu
# ./do_tagprop.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu
# ./do_tagfeat.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

train_data=yfcc_rnd_tn
valid_data=yfcc_rnd_vd
# rm -f $SURVEY_DB/*knn*
# ./do_knntagrel.sh ${train_data} ${valid_data} vgg-verydeep-16-fc7relu

# rm -f $SURVEY_DB/*tagvote*
# ./do_tagvote.sh ${train_data} ${valid_data} vgg-verydeep-16-fc7relu

# wordnet_frequency_tags._min_freq
./do_getknn.sh ${train_data} ${train_data} vgg-verydeep-16-fc7relu 0 1 1
./do_tagprop.sh ${train_data} ${valid_data} vgg-verydeep-16-fc7relu


