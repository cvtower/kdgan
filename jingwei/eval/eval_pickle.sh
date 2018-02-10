# ./eval_pickle.sh yfcc0k

# export BASEDIR=/Users/xiaojiew1/Projects # mac
export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan_xw/jingwei
export SURVEY_DB=$BASEDIR/kdgan_xw/results/runs
# export MATLAB_PATH=/Applications/MATLAB_R2017b.app/bin # mac
export MATLAB_PATH=/usr/local/bin
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 testCollection"
    exit
fi

testCollection=$1

export SURVEY_EVAL=$BASEDIR/kdgan_xw/results/eval
[ -d $SURVEY_EVAL ] || mkdir $SURVEY_EVAL
runfile=$SURVEY_EVAL/runs_"$testCollection".txt
resfile=$SURVEY_EVAL/runs_"$testCollection".res

# ls -d "$SURVEY_DB"/* > $runfile
ls -d "$SURVEY_DB"/* | grep kdgan > $runfile
# if [ -f "$resfile" ]; then
#     echo "result file exists at $resfile"
#     exit
# fi


if [ ! -f "$runfile" ]; then
    echo "runfile $runfile not exists!"
    exit
fi

if [ "$testCollection" == "flickr81" ]; then
    annotationName=concepts81.txt
elif [ "$testCollection" == "mirflickr08" ]; then
    annotationName=conceptsmir14.txt
elif [ "$testCollection" == "flickr55" -o "$testCollection" == "flickr51" ]; then
    annotationName=concepts51ms.txt
elif [ "$testCollection" == "yfcc0k" ]; then
    annotationName=concepts.txt
elif [ "$testCollection" == "yfcc1k" ]; then
    annotationName=concepts.txt
else
    echo "unknown testCollection $testCollection"
    exit
fi

# python $codepath/eval/eval_pickle.py $testCollection $annotationName $runfile > $resfile
python $codepath/eval/eval_pickle.py $testCollection $annotationName $runfile

