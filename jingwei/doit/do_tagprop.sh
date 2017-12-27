# ./do_tagprop.sh yfcc8k yfcc2k vgg-verydeep-16fc7relu

# export BASEDIR=/Users/xiaojiew1/Projects # mac
export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan/jingwei
export SURVEY_DB=$BASEDIR/kdgan/logs
# export MATLAB_PATH=/Applications/MATLAB_R2017b.app # mac
export MATLAB_PATH=/usr/local
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 trainCollection testCollection feature"
    exit
fi

trainCollection=$1
testCollection=$2
feature=$3
resultfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_$feature,tagprop.pkl

if [ "$feature" = "color64+dsift" ]; then
    distance=l1
elif [ "$feature" = "vgg-verydeep-16fc7relu" ]; then
    distance=cosine
else
    echo "unknown feature $feature"
    exit
fi

if [ "$testCollection" == "flickr81" ]; then
    testAnnotationName=concepts81.txt
elif [ "$testCollection" == "flickr51" ]; then
    testAnnotationName=concepts51ms.txt
elif [ "$testCollection" == "mirflickr08" ]; then
    testAnnotationName=conceptsmir14.txt
elif [ "$testCollection" == "yfcc2k" ]; then
    testAnnotationName=concepts.txt
else
    echo "unknown testCollection $testCollection"
    exit
fi

tagsh5file=$rootpath/$trainCollection/TextData/lemm_wordnet_freq_tags.h5
if [ ! -f "$tagsh5file" ]; then
    cd $rootpath/${trainCollection}
    python $codepath/tools/wordnet_frequency_tags.py 
    cd -
fi

python $codepath/model_based/tagprop/prepare_tagprop_data.py --distance $distance ${testCollection} ${trainCollection} $testAnnotationName $feature

# python $codepath/model_based/tagprop/tagprop.py --distance $distance ${testCollection} ${trainCollection} $testAnnotationName $feature $resultfile
