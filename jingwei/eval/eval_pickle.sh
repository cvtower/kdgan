# ./eval_pickle.sh yfcc2k jingwei

# export BASEDIR=/Users/xiaojiew1/Projects # mac
export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan/jingwei
export SURVEY_DB=$BASEDIR/kdgan/logs
export MATLAB_PATH=/Applications/MATLAB_R2017b.app/bin
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 testCollection method"
    exit
fi

testCollection=$1
method=$2
runfile=$rootpath/eval_output/runs_"$method"_"$testCollection".txt
resfile=$rootpath/eval_output/runs_"$method"_"$testCollection".res

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
elif [ "$testCollection" == "yfcc2k" ]; then
    annotationName=concepts.txt
else
    echo "unknown testCollection $testCollection"
    exit
fi

python $codepath/eval/eval_pickle.py $testCollection $annotationName $runfile > $resfile
# python $codepath/eval/eval_pickle.py $testCollection $annotationName $runfile

