# ./do_tagprop.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu

# export BASEDIR=/Users/xiaojiew1/Projects # mac
export BASEDIR=/home/xiaojie/Projects
export SURVEY_DATA=$BASEDIR/data/yfcc100m/survey_data
export SURVEY_CODE=$BASEDIR/kdgan/jingwei
export SURVEY_DB=$BASEDIR/kdgan/results/runs
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

if [ "$feature" = "color64+dsift" ]; then
  distance=l1
elif [ "$feature" = "vgg-verydeep-16-fc7relu" ]; then
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
elif [ "$testCollection" == "yfcc0k" ]; then
  testAnnotationName=concepts.txt
elif [ "$testCollection" == "yfcc9k" ]; then
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

for k in 200 400 600 800 1000
do
  python $codepath/model_based/tagprop/prepare_tagprop_data.py \
      --distance $distance \
      --k $k \
      ${testCollection} ${trainCollection} $testAnnotationName $feature
  # continue
  for variant in rank dist ranksigmoids distsigmoids
  do
    resultfile=$SURVEY_DB/"$trainCollection"_"$testCollection"_$feature,tagprop,$variant,$k.pkl
    python $codepath/model_based/tagprop/tagprop.py \
        --distance $distance \
        --k $k \
        --variant $variant \
        ${testCollection} ${trainCollection} $testAnnotationName $feature $resultfile
  done
  # exit
done

