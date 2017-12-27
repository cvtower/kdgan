# ./do_tagvote.sh yfcc8k yfcc2k vgg-verydeep-16fc7relu

export BASEDIR=/Users/xiaojiew1/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan/jingwei
export SURVEY_DB=$BASEDIR/kdgan/logs
export MATLAB_PATH=/Applications/MATLAB_R2017b.app/bin
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
k=1000
tagger=tagvote


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

annotationName=concepts.txt
python $codepath/instance_based/apply_tagger.py $testCollection $trainCollection $annotationName $feature --tagger $tagger --distance $distance --k $k
tagvotesfile=$rootpath/$testCollection/autotagging/$testCollection/$trainCollection/$annotationName/$tagger/$feature,"$distance"knn,$k/id.tagvotes.txt

if [ ! -f "$tagvotesfile" ]; then
    echo "tagvotes file $tagvotesfile does not exist!"
    exit
fi

conceptfile=$rootpath/$testCollection/Annotations/$testAnnotationName
resultfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_$feature,tagvote.pkl
python $codepath/postprocess/pickle_tagvotes.py $conceptfile $tagvotesfile $resultfile

