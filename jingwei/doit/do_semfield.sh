# ./do_semfield.sh yfcc8k yfcc2k wns
# ./do_semfield.sh yfcc8k yfcc2k fcs
# ./do_semfield.sh yfcc8k yfcc2k avgcos

export BASEDIR=/Users/xiaojiew1/Projects # mac
# export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan/jingwei
export SURVEY_DB=$BASEDIR/kdgan/logs
# export MATLAB_PATH=/Applications/MATLAB_R2017b.app # mac
export MATLAB_PATH=/usr/local
export PYTHONPATH=$PYTHONPATH:$SURVEY_CODE

rootpath=$SURVEY_DATA
codepath=$SURVEY_CODE

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 trainCollection testCollection tagsimMethod"
    exit
fi

trainCollection=$1
testCollection=$2
tagsimMethod=$3

if [[ "$tagsimMethod" != "wns" ]] && [[ "$tagsimMethod" != "fcs" ]] && [[ "$tagsimMethod" != "avgcos" ]] && [[ "$tagsimMethod" != "mulcos" ]]; then
    echo "unknown tag similarity method $tagsimMethod"
    exit
fi

vobfile=$rootpath/$trainCollection/TextData/wn."$trainCollection".txt
tagfreqfile=$rootpath/$trainCollection/TextData/lemmtag.userfreq.imagefreq.txt
jointfreqfile=$rootpath/$trainCollection/TextData/ucij.uuij.icij.iuij.txt

for datafile in $vobfile $tagfreqfile $jointfreqfile
do
    if [ ! -f "$datafile" ]; then
        echo "$datafile does not exist"
        exit
    fi
done


python $codepath/instance_based/dosemtagrel.py $testCollection $trainCollection $tagsimMethod

if [ "$testCollection" = "flickr51" ]; then
    annotationName=concepts51ms.txt
elif [ "$testCollection" == "flickr81" ]; then
    annotationName=concepts81.txt
elif [ "$testCollection" == "yfcc2k" ]; then
    annotationName=concepts.txt
else
    exit
fi



conceptfile=$rootpath/$testCollection/Annotations/$annotationName
tagvotesfile=$rootpath/$testCollection/tagrel/$testCollection/$trainCollection/$tagsimMethod-wn/id.tagvotes.txt
resultfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_semfield.pkl
if [ ! -f "$tagvotesfile" ]; then
    echo "$tagvotesfile does not exist"
    exit
fi

python $codepath/postprocess/pickle_tagvotes.py $conceptfile $tagvotesfile $resultfile

