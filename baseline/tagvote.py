# setup environment variables
import config

# create a tagvote instance
from instance_based.tagvote import TagVoteTagger
trainCollection = 'train10k'
annotationName = 'concepts130.txt'
feature = 'vgg-verydeep-16-fc7relu'
tagger = TagVoteTagger(collection=trainCollection, annotationName=annotationName, feature=feature, distance='cosine')

# open feature file of mirflickr08
from basic.constant import ROOT_PATH
from util.simpleknn.bigfile import BigFile
import os
rootpath = ROOT_PATH
testCollection = 'mirflickr08'
feat_dir = os.path.join(rootpath, testCollection, 'FeatureData', feature)
feat_file = BigFile(feat_dir)

# load image ids of mirflickr08
from basic.util import readImageSet
testimset = readImageSet(testCollection)

# load a subset of 200 images for test
import random
testimset = random.sample(testimset, 200)
renamed, vectors = feat_file.read(testimset)

# perform tag relevance learning on the test set
import time
s_time = time.time()
results = [tagger.predict(vec) for vec in vectors]
timespan = time.time() - s_time
print ('processing %d images took %g seconds' % (len(renamed), timespan))

# evaluation
from basic.annotationtable import readConcepts, readAnnotationsFrom
testAnnotationName = 'conceptsmir14.txt'
concepts = readConcepts(testCollection, testAnnotationName)
print('{} concepts in mirflickr08'.format(len(concepts)))
nr_of_concepts = len(concepts)
label2imset = {}
im2labelset = {}
for i,concept in enumerate(concepts):
    names,labels = readAnnotationsFrom(testCollection, testAnnotationName, concept)
    pos_set = [x[0] for x in zip(names,labels) if x[1]>0]
    print ('%s has %d positives' % (concept, len(pos_set)))
    for im in pos_set:
        label2imset.setdefault(concept, set()).add(im)
        im2labelset.setdefault(im, set()).add(concept)

# sort images to compute AP scores per concept
ranklists = {}
for _id, res in zip(renamed,results):
    for tag,score in res:
        ranklists.setdefault(tag, []).append((_id, score))
from basic.metric import getScorer
scorer = getScorer('AP')
mean_ap = 0.0
for i,concept in enumerate(concepts):
    pos_set = label2imset[concept]
    ranklist = ranklists[concept]
    # sort images by scores in descending order
    ranklist.sort(key=lambda v:(v[1], v[0]), reverse=True)
    sorted_labels = [2*int(x[0] in pos_set)-1 for x in ranklist]
    perf = scorer.score(sorted_labels)
    print ('%s %.3f' % (concept, perf))
    mean_ap += perf
mean_ap /= len(concepts)
print ('meanAP %.3f' % mean_ap)

# compute iAP per image
miap = 0.0
for _id, res in zip(renamed,results):
    # some images might be negatives to all the 14 concepts
    pos_set = im2labelset.get(_id, set())
    # evaluate only concepts with ground truth
    ranklist = [x for x in res if x[0] in label2imset]
    sorted_labels = [2*int(x[0] in pos_set)-1 for x in ranklist]
    perf = scorer.score(sorted_labels)
    miap += perf
miap /= len(renamed)
print ('miap %.3f' % miap)


