# ./do_getknn.sh yfcc8k yfcc8k vgg-verydeep-16-fc7relu 0 1 1
# ./do_getknn.sh yfcc8k yfcc2k vgg-verydeep-16-fc7relu 0 1 1

# export BASEDIR=/Users/xiaojiew1/Projects # mac
export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan/jingwei
export SURVEY_DB=$BASEDIR/kdgan/runs
# export MATLAB_PATH=/Applications/MATLAB_R2017b.app/bin # mac
export MATLAB_PATH=/usr/local/bin
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 6 ]; then
    echo "Usage: $0 trainCollection testCollection feature uu numjobs job"
    exit
fi

trainCollection=$1
testCollection=$2
feature=$3

if [ "$feature" = "color64+dsift" ]; then
    distance=l1
elif [ "$feature" = "vgg-verydeep-16-fc7relu" ]; then 
    distance=cosine
else
    echo "unknown feature $feature"
    exit
fi 
uniqueUser=$4
numjobs=$5
job=$6

python $codepath/instance_based/getknn.py  $trainCollection $testCollection $feature --distance $distance --uu $uniqueUser --numjobs $numjobs --job $job

