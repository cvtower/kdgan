# setup environment variables
import config

# create a tagvote instance
from instance_based.tagvote import TagVoteTagger
trainCollection = 'yfcc8k'
annotationName = 'concepts.txt'
feature = 'vgg-verydeep-16fc7relu'
tagger = TagVoteTagger(collection=trainCollection, annotationName=annotationName, feature=feature, distance='cosine')

# open feature file of valid data
from basic.constant import ROOT_PATH
from util.simpleknn.bigfile import BigFile
import os
rootpath = ROOT_PATH
testCollection = 'yfcc2k'
feat_dir = os.path.join(rootpath, testCollection, 'FeatureData', feature)
feat_file = BigFile(feat_dir)

# load image ids of valid data
from basic.util import readImageSet
testimset = readImageSet(testCollection)

# load a subset of 10 images for test
# import random
# testimset = random.sample(testimset, 10)

# load valid images for test
renamed, vectors = feat_file.read(testimset)

# perform tag relevance learning on the test set
import time
s_time = time.time()
results = [tagger.predict(vec) for vec in vectors]
timespan = time.time() - s_time
print ('processing %d images took %g seconds' % (len(renamed), timespan))

import operator
from os import path
import utils
utils.create_if_nonexist(config.logs_dir)
outfile = path.join(config.logs_dir, 'tagvote.res')
fout = open(outfile, 'w')
for _id,res in zip(renamed,results):
    res = sorted(res, key=operator.itemgetter(1), reverse=True)
    fout.write('{0}'.format(_id))
    for i in range(10):
        tag,score = res[i]
        fout.write('\t{0}:{1:.2f}'.format(tag, score))
    fout.write('\n')
fout.close()


