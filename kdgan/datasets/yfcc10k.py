from kdgan import config, utils
from datasets import imagenet

import operator
import os
import random
import shutil
import urllib

from os import path
from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

SPACE_PLACEHOLDER = '+'
FIELD_SEPERATOR = '\t'
LABEL_SEPERATOR = ','

NUM_FIELD = 6

POST_INDEX = 0
USER_INDEX = 1
IMAGE_INDEX = 2
TEXT_INDEX = 3
DESC_INDEX = 4
LABEL_INDEX = -1

NUM_TOP_LABEL = 100 # select top 100 labels
EXPECTED_NUM_POST = 10000
MIN_IMAGE_PER_USER = 10
MAX_IMAGE_PER_USER = 100
MIN_IMAGE_PER_LABEL = 100
POST_UNIT_SIZE = 5
TRAIN_RATIO = 0.80

def create_if_nonexist(outdir):
    if not path.exists(outdir):
        os.makedirs(outdir)

def skip_if_exist(filepath):
    skip = False
    if path.isfile(filepath):
        skip = True
    return skip

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

def check_num_field():
    fin = open(config.sample_filepath)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        num_field = len(fields)
        if num_field != NUM_FIELD:
            raise Exception('wrong number of fields')
    fin.close()

def select_top_label():
    valid = set()
    label_names = imagenet.create_readable_names_for_imagenet_labels()
    for label in range(1, 1001):
        names = label_names[label].split(',')
        for name in names:
            name = name.strip().lower()
            name = name.replace(' ', SPACE_PLACEHOLDER)
            word = name.split(SPACE_PLACEHOLDER)[-1]
            valid.add(name)
            valid.add(word)

    fin = open(config.sample_filepath)
    label_count = {}
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        labels = fields[LABEL_INDEX]

        labels = labels.split(LABEL_SEPERATOR)
        for label in labels:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
    fin.close()

    invalid = ['bank', 'center', 'engine', 'home', 'jack', 'maria', 'train', ]
    label_count = sorted(label_count.items(), key=operator.itemgetter(1, 0), reverse=True)

    top_labels, num_label, = set(), 0
    for label, _ in label_count:
        if num_label == NUM_TOP_LABEL:
            break
        if label in invalid:
            continue
        if label not in valid:
            continue
        top_labels.add(label)
        num_label += 1
    utils.save_collection(top_labels, config.label_filepath)

def select_posts():
    top_labels = utils.load_collection(config.label_filepath)
    user_posts = {}
    fin = open(config.sample_filepath)
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

    label_count = {}
    for user, posts in user_posts.items():
        for post in posts:
            fields = post.split(FIELD_SEPERATOR)
            user = fields[USER_INDEX]
            labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
            for label in labels:
                if label not in label_count:
                    label_count[label] = 0
                label_count[label] += 1

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

    users = sorted(user_posts.keys())
    for user in users:
        keep = False
        posts = user_posts[user]
        user_label_count = {}
        for post in posts:
            fields = post.split(FIELD_SEPERATOR)
            labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
            for label in labels:
                if label not in user_label_count:
                    user_label_count[label] = 0
                user_label_count[label] += 1
        for label, count in user_label_count.items():
            if (label_count[label] - user_label_count[label]) < MIN_IMAGE_PER_LABEL:
                keep = True
                break
        if not keep:
            num_post = len(posts)
            if dif_post - num_post < 0:
                if num_post - dif_post < MIN_IMAGE_PER_USER:
                    continue
                else:
                    num_post -= dif_post
                    user_posts[user] = posts[:num_post]
                    break
                print('todo')
                exit()
            else:
                for post in posts:
                    fields = post.split(FIELD_SEPERATOR)
                    labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
                    for label in labels:
                        label_count[label] -= 1
                dif_post -= num_post
                del user_posts[user]
    tot_post = count_posts(user_posts)
    user_count = {}
    for user, posts in user_posts.items():
        user_count[user] = len(posts)
    dif_post = tot_post - EXPECTED_NUM_POST
    print('{} to be removed'.format(dif_post))

    users = sorted(user_posts.keys())
    sorted_user_count = sorted(user_count.items(), key=operator.itemgetter(1), reverse=True)
    user_posts_cpy = user_posts
    user_posts = {}
    for user in users:
        posts = user_posts_cpy[user]
        keep_posts = []
        disc_posts = []
        for post in posts:
            keep = False
            fields = post.split(FIELD_SEPERATOR)
            labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
            for label in labels:
                if (label_count[label] - 1) < MIN_IMAGE_PER_LABEL:
                    keep = True
                    break
            if keep:
                keep_posts.append(post)
            else:
                disc_posts.append(post)
                for label in labels:
                    label_count[label] -= 1

        num_keep = max(len(keep_posts), MIN_IMAGE_PER_USER)
        if num_keep % POST_UNIT_SIZE != 0:
            num_keep = (num_keep // POST_UNIT_SIZE + 1) * POST_UNIT_SIZE

        if dif_post == 0:
            num_keep = len(posts)
        else:
            num_disc = len(posts) - num_keep
            if dif_post - num_disc < 0:
                num_keep += (num_disc - dif_post)
                dif_post = 0
            else:
                dif_post -= num_disc

        num_rest = num_keep - len(keep_posts)
        for post in disc_posts[:num_rest]:
            keep_posts.append(post)
            fields = post.split(FIELD_SEPERATOR)
            labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
            for label in labels:
                label_count[label] += 1
        disc_posts = disc_posts[num_rest:]
        user_posts[user] = keep_posts
    print('{} to be removed'.format(dif_post))

    user_posts_cpy = user_posts
    user_posts = {}
    for user, posts in user_posts_cpy.items():
        user_posts[user] = []
        for post in posts:
            fields = post.split(FIELD_SEPERATOR)
            filename = fields[IMAGE_INDEX]
            image = filename.split('_')[0]
            fields[IMAGE_INDEX] = image
            post = FIELD_SEPERATOR.join(fields)
            user_posts[user].append(post)
    save_posts(user_posts, config.raw_filepath)

def tokenize_dataset():
    stemmer = SnowballStemmer('english')
    stopwords = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer('[a-z]+')
    def _in_synsets(token):
        if wordnet.synsets(token):
            return True
        else:
            return False
    def _tokenize(tokens):
        tokens = [token for token in tokens if _in_synsets(token)]
        tokens = [token for token in tokens if not token in stopwords]
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens

    fin = open(config.raw_filepath)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        text = fields[TEXT_INDEX]
        desc = fields[DESC_INDEX]
        text = ' '.join([text, desc])

        text = urllib.parse.unquote(text)
        text = text.replace(SPACE_PLACEHOLDER, ' ')

        soup = BeautifulSoup(text, 'html.parser')
        children = []
        for child in soup.children:
            if type(child) != NavigableString:
                continue
            children.append(str(child))
        text = ' '.join(children)

        tokens = word_tokenize(text)
        tokens = _tokenize(tokens)
        if len(tokens) == 0:
            tokens = tokenizer.tokenize(text)
            tokens = _tokenize(tokens)
        text = ' '.join(tokens)
        print(text)

    fin.close()

def check_dataset(infile):
    top_labels = utils.load_collection(config.label_filepath)
    top_labels = set(top_labels)
    fin = open(infile)
    while True:
        line = fin.readline()
        if not line:
            break
        fields = line.strip().split(FIELD_SEPERATOR)
        labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
        for label in labels:
            top_labels.discard(label)
    assert len(top_labels) == 0

def split_dataset():
    user_posts = {}
    fin = open(config.raw_filepath)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        user = fields[USER_INDEX]
        if user not in user_posts:
            user_posts[user] = []
        user_posts[user].append(line)
    fin.close()
    train_user_posts = {}
    valid_user_posts = {}
    for user, posts in user_posts.items():
        num_post = len(posts)
        if (num_post % POST_UNIT_SIZE) == 0:
            seperator = int(num_post * TRAIN_RATIO)
        else:
            seperator = int(num_post * TRAIN_RATIO) + 1
        train_user_posts[user] = posts[:seperator]
        valid_user_posts[user] = posts[seperator:]

    save_posts(train_user_posts, config.train_filepath)
    save_posts(valid_user_posts, config.valid_filepath)

    check_dataset(config.train_filepath)
    check_dataset(config.valid_filepath)

def count_dataset():
    create_if_nonexist(config.temp_dir)
    user_count = {}
    fin = open(config.raw_filepath)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        user = fields[USER_INDEX]
        if user not in user_count:
            user_count[user] = 0
        user_count[user] += 1
    fin.close()
    sorted_user_count = sorted(user_count.items(), key=operator.itemgetter(0), reverse=True)
    outfile = path.join(config.temp_dir, 'user_count')
    with open(outfile, 'w') as fout:
        for user, count in sorted_user_count:
            fout.write('{}\t{}\n'.format(user, count))

    label_count = {}
    fin = open(config.raw_filepath)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        user = fields[USER_INDEX]
        labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
        assert len(labels) != 0
        for label in labels:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
    fin.close()
    sorted_label_count = sorted(label_count.items(), key=operator.itemgetter(0), reverse=True)
    outfile = path.join(config.temp_dir, 'label_count')
    labels, lemms = set(), set()
    with open(outfile, 'w') as fout:
        for label, count in sorted_label_count:
            labels.add(label)
            lemm = lemmatizer.lemmatize(label)
            lemms.add(lemm)
            if lemm != label:
                print('{}->{}'.format(lemm, label))
            fout.write('{}\t{}\n'.format(label, count))
    print('#label={} #lemm={}'.format(len(labels), len(lemms)))

if __name__ == '__main__':
    create_if_nonexist(config.yfcc10k_dir)
    check_num_field()
    if not skip_if_exist(config.label_filepath):
        print('select top labels')
        select_top_label()
    if not skip_if_exist(config.raw_filepath):
        print('select posts')
        select_posts()

    # if not skip_if_exist(config.train_filepath) or not skip_if_exist(config.valid_filepath):
    #     print('split dataset')
    #     split_dataset()


