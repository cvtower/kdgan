from kdgan import config, utils

import operator
import os
import random
import shutil
import urllib

import numpy as np
import tensorflow as tf

from datasets import dataset_utils
from datasets import imagenet
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from os import path
from tensorflow.contrib import slim

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from datasets.download_and_convert_flowers import ImageReader
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from PIL import Image

tf.app.flags.DEFINE_boolean('dev', False, '')
tf.app.flags.DEFINE_string('model_name', None, '')
tf.app.flags.DEFINE_string('preprocessing_name', None, '')
tf.app.flags.DEFINE_string('checkpoint_path', None, '')
tf.app.flags.DEFINE_string('end_point', None, '')
tf.app.flags.DEFINE_integer('num_epoch', 250, '')
flags = tf.app.flags.FLAGS

lemmatizer = WordNetLemmatizer()

SPACE_PLUS = '+'
FIELD_SEPERATOR = '\t'
LABEL_SEPERATOR = ','

EXPECTED_NUM_FIELD = 6

POST_INDEX = 0
USER_INDEX = 1
IMAGE_INDEX = 2
TEXT_INDEX = 3
DESC_INDEX = 4
LABEL_INDEX = -1

NUM_TOP_LABEL = 100 # select top 100 labels
EXPECTED_NUM_POST = 20000
MIN_IMAGE_PER_USER = 10
MAX_IMAGE_PER_USER = 100
MIN_IMAGE_PER_LABEL = 100
POST_UNIT_SIZE = 10
TRAIN_RATIO = 0.90

def check_num_field():
  print('check %s' % path.basename(config.sample_file))
  fin = open(config.sample_file)
  while True:
    line = fin.readline()
    if not line:
      print('line=\'{}\' type={}'.format(line, type(line)))
      break
    fields = line.strip().split(FIELD_SEPERATOR)
    num_field = len(fields)
    if num_field != EXPECTED_NUM_FIELD:
      raise Exception('wrong number of fields')
  fin.close()

def select_top_label():
  imagenet_labels = {}
  label_names = imagenet.create_readable_names_for_imagenet_labels()
  label_names = {k:v.lower() for k, v in label_names.items()}
  for label_id in range(1, 1001):
    names = label_names[label_id]
    for name in names.split(','):
      name = name.strip()
      label = name.split()[-1]
      if label not in imagenet_labels:
        imagenet_labels[label] = []
      imagenet_labels[label].append(names)

  fin = open(config.sample_file)
  label_count = {}
  while True:
    line = fin.readline().strip()
    if not line:
      break
    fields = line.split(FIELD_SEPERATOR)
    labels = fields[LABEL_INDEX]
    labels = labels.split(LABEL_SEPERATOR)
    for label in labels:
      label_count[label] = label_count.get(label, 0) + 1
  fin.close()

  invalid_labels = ['bank', 'center', 'home', 'jack', 'maria']
  invalid_labels.append('apple') # custard apple
  invalid_labels.append('bar') # horizontal bar, high bar
  invalid_labels.append('coral') # brain coral
  invalid_labels.append('dragon') # komodo dragon
  invalid_labels.append('grand') # grand piano
  invalid_labels.append('star') # starfish, sea star
  label_count = sorted(label_count.items(),
      key=operator.itemgetter(1, 0),
      reverse=True)

  top_labels, num_label, = set(), 0
  for label, _ in label_count:
    if num_label == NUM_TOP_LABEL:
      break
    if label in invalid_labels:
      continue
    if label not in imagenet_labels:
      continue
    top_labels.add(label)
    num_label += 1
  top_labels = sorted(top_labels)
  for label in top_labels:
    names = []
    for label_id in range(1, 1001):
      if label in label_names[label_id]:
        names.append(label_names[label_id])
    print('label=%s' % (label))
    for names in imagenet_labels[label]:
      print('\t%s' %(names))
  utils.save_collection(top_labels, config.label_file)

def with_top_label(labels, top_labels):
    old_labels = labels.split(LABEL_SEPERATOR)
    new_labels = []
    for label in old_labels:
        if label not in top_labels:
            continue
        new_labels.append(label)
    if len(new_labels) == 0:
        return False
    return True

def keep_top_label(labels, top_labels):
    old_labels = labels.split(LABEL_SEPERATOR)
    new_labels = []
    for label in old_labels:
        if label not in top_labels:
            continue
        new_labels.append(label)
    return new_labels

def count_posts(user_posts):
    tot_post = 0
    for user, posts in user_posts.items():
        tot_post += len(posts)
    return tot_post

def save_posts(user_posts, infile):
    image_set = set()
    with open(infile, 'w') as fout:
        users = sorted(user_posts.keys())
        for user in users:
            posts = user_posts[user]
            posts = sorted(posts, key=operator.itemgetter(0))
            for post in posts:
                fields = post.split(FIELD_SEPERATOR)
                image = fields[IMAGE_INDEX]
                image_set.add(image)
                fout.write('%s\n' % post)
    print('#image={}'.format(len(image_set)))

def select_posts():
    top_labels = utils.load_collection(config.label_file)
    user_posts = {}
    fin = open(config.sample_file)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        user, labels = fields[USER_INDEX], fields[LABEL_INDEX]
        image_url = fields[IMAGE_INDEX]
        if not with_top_label(labels, top_labels):
            continue
        labels = keep_top_label(labels, top_labels)
        fields[LABEL_INDEX] = LABEL_SEPERATOR.join(labels)
        fields[IMAGE_INDEX] = path.basename(image_url)
        if user not in user_posts:
            user_posts[user] = []
        user_posts[user].append(FIELD_SEPERATOR.join(fields))
    fin.close()

    user_posts_cpy = user_posts
    user_posts = {}
    for user in user_posts_cpy.keys():
        posts = user_posts_cpy[user]
        num_post = len(posts)
        if num_post < MIN_IMAGE_PER_USER:
            continue
        user_posts[user] = posts
    tot_post = count_posts(user_posts)
    print('#post=%d' % (tot_post))

    label_count = {}
    for user, posts in user_posts.items():
        for post in posts:
            fields = post.split(FIELD_SEPERATOR)
            user = fields[USER_INDEX]
            labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
            for label in labels:
                label_count[label] = label_count.get(label, 0) + 1
    counts = label_count.values()
    print('min=%d max=%d' % (min(counts), max(counts)))

    users = sorted(user_posts.keys())
    user_posts_cpy = user_posts
    user_posts = {}
    for user in users:
        posts = user_posts_cpy[user]
        num_post = len(posts)
        num_post = min(num_post // POST_UNIT_SIZE * POST_UNIT_SIZE, MAX_IMAGE_PER_USER)
        not_keep = len(posts) - num_post
        user_posts[user] = []
        count = 0
        for post in posts:
            keep = False
            fields = post.split(FIELD_SEPERATOR)
            labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
            for label in labels:
                if (label_count[label] - 1) < MIN_IMAGE_PER_LABEL:
                    keep = True
                    break
            if count >= not_keep:
                user_posts[user].append(post)
            else:
                if keep:
                    user_posts[user].append(post)
                else:
                    count += 1
                    for label in labels:
                        label_count[label] -= 1
        posts = user_posts[user]
        num_post = len(user_posts[user])
        if (num_post // POST_UNIT_SIZE) != 0:
            num_post = num_post // POST_UNIT_SIZE * POST_UNIT_SIZE
            user_posts[user] = posts[:num_post]
            for post in posts[num_post:]:
                fields = post.split(FIELD_SEPERATOR)
                labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
                for label in labels:
                    label_count[label] -= 1
    tot_post = count_posts(user_posts)
    dif_post = tot_post - EXPECTED_NUM_POST
    print('{} to be removed'.format(dif_post))
    print('#user=%d' % (len(user_posts)))

    # users = sorted(user_posts.keys())
    # for user in users:
    #     keep = False
    #     posts = user_posts[user]
    #     user_label_count = {}
    #     for post in posts:
    #         fields = post.split(FIELD_SEPERATOR)
    #         labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
    #         for label in labels:
    #             if label not in user_label_count:
    #                 user_label_count[label] = 0
    #             user_label_count[label] += 1
    #     for label, count in user_label_count.items():
    #         if (label_count[label] - user_label_count[label]) < MIN_IMAGE_PER_LABEL:
    #             keep = True
    #             break
    #     if not keep:
    #         num_post = len(posts)
    #         if dif_post - num_post < 0:
    #             if num_post - dif_post < MIN_IMAGE_PER_USER:
    #                 continue
    #             else:
    #                 num_post -= dif_post
    #                 user_posts[user] = posts[:num_post]
    #                 break
    #             print('todo')
    #             exit()
    #         else:
    #             for post in posts:
    #                 fields = post.split(FIELD_SEPERATOR)
    #                 labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
    #                 for label in labels:
    #                     label_count[label] -= 1
    #             dif_post -= num_post
    #             del user_posts[user]
    # tot_post = count_posts(user_posts)
    # user_count = {}
    # for user, posts in user_posts.items():
    #     user_count[user] = len(posts)
    # dif_post = tot_post - EXPECTED_NUM_POST
    # print('{} to be removed'.format(dif_post))

    # users = sorted(user_posts.keys())
    # sorted_user_count = sorted(user_count.items(), key=operator.itemgetter(1), reverse=True)
    # user_posts_cpy = user_posts
    # user_posts = {}
    # for user in users:
    #     posts = user_posts_cpy[user]
    #     keep_posts = []
    #     disc_posts = []
    #     for post in posts:
    #         keep = False
    #         fields = post.split(FIELD_SEPERATOR)
    #         labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
    #         for label in labels:
    #             if (label_count[label] - 1) < MIN_IMAGE_PER_LABEL:
    #                 keep = True
    #                 break
    #         if keep:
    #             keep_posts.append(post)
    #         else:
    #             disc_posts.append(post)
    #             for label in labels:
    #                 label_count[label] -= 1

    #     num_keep = max(len(keep_posts), MIN_IMAGE_PER_USER)
    #     if num_keep % POST_UNIT_SIZE != 0:
    #         num_keep = (num_keep // POST_UNIT_SIZE + 1) * POST_UNIT_SIZE

    #     if dif_post == 0:
    #         num_keep = len(posts)
    #     else:
    #         num_disc = len(posts) - num_keep
    #         if dif_post - num_disc < 0:
    #             num_keep += (num_disc - dif_post)
    #             dif_post = 0
    #         else:
    #             dif_post -= num_disc

    #     num_rest = num_keep - len(keep_posts)
    #     for post in disc_posts[:num_rest]:
    #         keep_posts.append(post)
    #         fields = post.split(FIELD_SEPERATOR)
    #         labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
    #         for label in labels:
    #             label_count[label] += 1
    #     disc_posts = disc_posts[num_rest:]
    #     user_posts[user] = keep_posts
    # print('{} to be removed'.format(dif_post))

    # user_posts_cpy = user_posts
    # user_posts = {}
    # for user, posts in user_posts_cpy.items():
    #     user_posts[user] = []
    #     for post in posts:
    #         fields = post.split(FIELD_SEPERATOR)
    #         filename = fields[IMAGE_INDEX]
    #         image = filename.split('_')[0]
    #         fields[IMAGE_INDEX] = image
    #         post = FIELD_SEPERATOR.join(fields)
    #         user_posts[user].append(post)

    # save_posts(user_posts, config.raw_file)

def main(_):
    check_num_field()
    utils.create_if_nonexist(config.dataset_dir)
    if not utils.skip_if_exist(config.label_file):
        print('select top labels')
        select_top_label()
    if not utils.skip_if_exist(config.raw_file):
        print('select posts')
        select_posts()

if __name__ == '__main__':
  tf.app.run()