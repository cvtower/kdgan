from kdgan import config, utils
from datasets import imagenet

import operator
import os
import random
import shutil
import urllib

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from nltk import word_tokenize
from nltk.corpus import stopwords, wordnet
from os import path

from bs4 import BeautifulSoup
from bs4.element import NavigableString
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from PIL import Image

# tf slim
from datasets import dataset_utils
from datasets.download_and_convert_flowers import ImageReader

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
EXPECTED_NUM_POST = 10000
MIN_IMAGE_PER_USER = 10
MAX_IMAGE_PER_USER = 100
MIN_IMAGE_PER_LABEL = 100
POST_UNIT_SIZE = 5
TRAIN_RATIO = 0.80

def create_if_nonexist(outdir):
    if not path.exists(outdir):
        os.makedirs(outdir)

def skip_if_exist(infile):
    skip = False
    if path.isfile(infile):
        skip = True
    return skip

def check_num_field():
    fin = open(config.sample_file)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        num_field = len(fields)
        # print(num_field)
        if num_field != EXPECTED_NUM_FIELD:
            raise Exception('wrong number of fields')
    fin.close()

def select_top_label():
    imagenet_labels = set()
    label_names = imagenet.create_readable_names_for_imagenet_labels()
    for label in range(1, 1001):
        names = label_names[label].split(',')
        for name in names:
            name = name.strip().lower()
            name = name.replace(' ', SPACE_PLUS)
            word = name.split(SPACE_PLUS)[-1]
            imagenet_labels.add(name)
            imagenet_labels.add(word)

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
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
    fin.close()

    invalid_labels = ['bank', 'center', 'engine', 'home', 'jack', 'maria', 'train', ]
    label_count = sorted(label_count.items(), key=operator.itemgetter(1, 0), reverse=True)

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

    save_posts(user_posts, config.raw_file)

stopwords = set(stopwords.words('english'))    
def tokenize_dataset():
    stemmer = SnowballStemmer('english')
    tokenizer = RegexpTokenizer('[a-z]+')
    def _in_wordnet(token):
        if wordnet.synsets(token):
            return True
        else:
            return False
    def _stop_stem(tokens):
        tokens = [token for token in tokens if _in_wordnet(token)]
        tokens = [token for token in tokens if not token in stopwords]
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens

    fin = open(config.raw_file)
    fout = open(config.data_file, 'w')
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        post = fields[POST_INDEX]
        text = fields[TEXT_INDEX]
        desc = fields[DESC_INDEX]
        # print('{0}\n{1}\n{0}'.format('#'*80, post))
        # print(urllib.parse.unquote(text).replace(SPACE_PLUS, ' '))
        # print('{0}'.format('#'*80))
        # print(urllib.parse.unquote(desc).replace(SPACE_PLUS, ' '))
        # print('{0}'.format('#'*80))
        text = ' '.join([text, desc])

        text = urllib.parse.unquote(text)
        text = text.replace(SPACE_PLUS, ' ')

        soup = BeautifulSoup(text, 'html.parser')
        children = []
        for child in soup.children:
            if type(child) == NavigableString:
                children.append(str(child))
            else:
                children.append(str(child.text))
        text = ' '.join(children)

        tokens = word_tokenize(text)
        tokens = _stop_stem(tokens)
        if len(tokens) == 0:
            tokens = tokenizer.tokenize(text)
            tokens = _stop_stem(tokens)
        text = ' '.join(tokens)
        # print(text)
        # print('{0}\n{0}'.format('#'*80))
        labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
        labels = ' '.join(labels)
        fields = [
            fields[POST_INDEX],
            fields[USER_INDEX],
            fields[IMAGE_INDEX],
            text,
            labels,
        ]
        fout.write('%s\n' % FIELD_SEPERATOR.join(fields))
    fout.close()
    fin.close()

def check_dataset(infile):
    top_labels = utils.load_collection(config.label_file)
    top_labels = set(top_labels)
    fin = open(infile)
    while True:
        line = fin.readline()
        if not line:
            break
        fields = line.strip().split(FIELD_SEPERATOR)
        labels = fields[LABEL_INDEX].split()
        for label in labels:
            top_labels.discard(label)
    # print(path.basename(infile), len(top_labels))
    assert len(top_labels) == 0

def split_dataset():
    user_posts = {}
    fin = open(config.data_file)
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

    save_posts(train_user_posts, config.train_file)
    save_posts(valid_user_posts, config.valid_file)

    check_dataset(config.train_file)
    check_dataset(config.valid_file)

    vocab = set()
    for user, posts in train_user_posts.items():
        for post in posts:
            fields = post.split(FIELD_SEPERATOR)
            tokens = fields[TEXT_INDEX].split()
            for token in tokens:
                vocab.add(token)
    vocab = sorted(vocab)
    if config.unk_token in vocab:
        print('please change unk token')
        exit()
    vocab.insert(0, config.unk_token)
    if config.pad_token in vocab:
        print('please change pad token')
        exit()
    vocab.insert(0, config.pad_token)
    utils.save_collection(vocab, config.vocab_file)

def count_dataset():
    create_if_nonexist(config.temp_dir)
    user_count = {}
    fin = open(config.data_file)
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
    fin = open(config.data_file)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        user = fields[USER_INDEX]
        labels = fields[LABEL_INDEX].split()
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

def get_image_path(image_dir, image_url):
    fields = image_url.split('/')
    image_path = path.join(image_dir, fields[-2], fields[-1])
    return image_path

def collect_image(infile, outdir):
    post_image = {}
    fin = open(infile)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        post = fields[POST_INDEX]
        image = fields[IMAGE_INDEX]
        post_image[post] = image
    fin.close()
    create_if_nonexist(outdir)
    fin = open(config.sample_file)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        post = fields[POST_INDEX]
        if post not in post_image:
            continue
        image_url = fields[IMAGE_INDEX]
        src_file = get_image_path(config.image_dir, image_url)
        image = post_image[post]
        dst_file = path.join(outdir, '%s.jpg' % image)
        if path.isfile(dst_file):
            continue
        shutil.copyfile(src_file, dst_file)

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def build_example(user, image, text, label_vec, extension, height, width, image_file):
    return tf.train.Example(features=tf.train.Features(feature={
        config.user_key:dataset_utils.bytes_feature(user),
        config.image_encoded_key:dataset_utils.bytes_feature(image),
        config.text_key:int64_feature(text),
        config.label_key:int64_feature(label_vec),
        config.image_format_key:dataset_utils.bytes_feature(extension),
        config.image_height_key:int64_feature([height]),
        config.image_width_key:int64_feature([width]),
        config.image_file_key:dataset_utils.bytes_feature(image_file),
    }))

def create_tfrecord(infile):
    tfrecord_file = '%s.tfrecord' % infile
    print(tfrecord_file)

    user_list = []
    image_list = []
    text_list = []
    label_list = []
    fin = open(infile)
    while True:
        line = fin.readline()
        if not line:
            break
        fields = line.strip().split(FIELD_SEPERATOR)
        user = fields[USER_INDEX]
        image = fields[IMAGE_INDEX]
        image_file = path.join(config.image_data_dir, '%s.jpg' % image)
        tokens = fields[TEXT_INDEX].split()
        labels = fields[LABEL_INDEX].split()
        user_list.append(user)
        image_list.append(image_file)
        text_list.append(tokens)
        label_list.append(labels)
    fin.close()

    label_to_id = utils.load_label_to_id()
    num_label = len(label_to_id)
    print('#label={}'.format(num_label))
    token_to_id = utils.load_token_to_id()
    unk_token_id = token_to_id[config.unk_token]
    vocab_size = len(token_to_id)
    print('#vocab={}'.format(vocab_size))

    reader = ImageReader()
    with tf.Session() as sess:
        with tf.python_io.TFRecordWriter(tfrecord_file) as fout:
            for user, image_file, text, labels in zip(user_list, image_list, text_list, label_list):
                user = bytes(user, encoding='utf-8')
                
                image = tf.gfile.FastGFile(image_file, 'rb').read()

                label_ids = [label_to_id[label] for label in labels]
                label_vec = np.zeros((num_label,), dtype=np.int64)
                label_vec[label_ids] = 1
                
                text = [token_to_id.get(token, unk_token_id) for token in text]

                extension = b'jpg'
                height, width = reader.read_image_dims(sess, image)
                image_file = bytes(image_file, encoding='utf-8')

                example = build_example(user, image, text, label_vec, extension, height, width, image_file)
                fout.write(example.SerializeToString())

def check_tfrecord(tfrecord_file, is_training):
    data_sources = [tfrecord_file]

    id_to_label = utils.load_id_to_label()
    num_label = len(id_to_label)
    print('#label={}'.format(num_label))
    id_to_token = utils.load_id_to_token()
    token_to_id = utils.load_token_to_id()
    unk_token_id = token_to_id[config.unk_token]
    vocab_size = int((len(id_to_token) + len(token_to_id)) / 2)
    print('#vocab={}'.format(vocab_size))

    reader = tf.TFRecordReader
    keys_to_features = {
        config.user_key:tf.FixedLenFeature((), tf.string, default_value=''),
        config.image_encoded_key:tf.FixedLenFeature((), tf.string, default_value=''),
        config.text_key:tf.VarLenFeature(dtype=tf.int64),
        config.label_key:tf.FixedLenFeature([num_label], tf.int64, default_value=tf.zeros([num_label], dtype=tf.int64)),
        config.image_format_key:tf.FixedLenFeature((), tf.string, default_value='jpg'),
        config.image_file_key:tf.FixedLenFeature((), tf.string, default_value='')
    }
    items_to_handlers = {
        'user':slim.tfexample_decoder.Tensor(config.user_key),
        'image':slim.tfexample_decoder.Image(),
        'text':slim.tfexample_decoder.Tensor(config.text_key, default_value=unk_token_id),
        'label':slim.tfexample_decoder.Tensor(config.label_key),
        'image_file':slim.tfexample_decoder.Tensor(config.image_file_key),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    num_samples = np.inf
    items_to_descriptions = {'user':'', 'image':'', 'text':'', 'label':'', 'image_file':'',}
    dataset = slim.dataset.Dataset(
        data_sources=data_sources,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=items_to_descriptions,
    )
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset, shuffle=is_training)
    user_ts, image_ts, text_ts, label_ts, image_file_ts = provider.get(
            ['user', 'image', 'text', 'label', 'image_file'])
    
    # with tf.Session() as sess:
    #     with slim.queues.QueueRunners(sess):
    #         for i in range(2):
    #             user_np, image_np, text_np, label_np, image_file_np = sess.run(
    #                     [user_ts, image_ts, text_ts, label_ts, image_file_ts])
    #             print('{0}\n{0}'.format('#'*80))
    #             print(user_np)
    #             Image.fromarray(np.asarray(image_np)).show()
    #             tokens = [id_to_token[text_np[i]] for i in range(text_np.shape[0])]
    #             print(tokens)
    #             label_ids = [i for i, l in enumerate(label_np) if l != 0]
    #             labels = [id_to_label[label_id] for label_id in label_ids]
    #             print(labels)
    #             print(image_file_np)
    #             print('{0}\n{0}'.format('#'*80))

    batch_size = 32
    num_step = 10000
    user_bt, image_bt, text_bt, label_bt, image_file_bt = tf.train.batch(
            [user_ts, image_ts, text_ts, label_ts, image_file_ts], 
            batch_size=batch_size, dynamic_pad=True)

    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            for i in range(num_step):
                user_np, image_np, text_np, label_np, image_file_np = sess.run(
                        [user_bt, image_bt, text_bt, label_bt, image_file_bt])
                # print(user_np.shape, image_np.shape, text_np.shape, label_np.shape, image_file_np.shape)
                for b in range(batch_size):
                    print(text_np[b,:])

def get_dataset(infile):
    datasize = len(open(infile).readlines())
    dataset = 'yfcc{}k'.format(datasize // 1000)
    return dataset

def survey_image_data(infile):
    dataset = get_dataset(infile)
    image_data = path.join(config.surv_dir, dataset, 'ImageData')
    create_if_nonexist(image_data)
    fout = open(path.join(image_data, '%s.txt' % dataset), 'w')
    fin = open(infile)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image = fields[IMAGE_INDEX]
        image_file = '%s.jpg' % image
        fout.write('{}\n'.format(image_file))
    fin.close()
    fout.close()
    collect_image(infile, image_data)

def survey_text_data(infile):
    seperator = '###'
    def _get_key(label_i, label_j):
        if label_i < label_j:
            key = label_i + seperator + label_j
        else:
            key = label_j + seperator + label_i
        return key
    def _get_labels(key):
        fields = key.split(seperator)
        label_i, label_j = fields[0], fields[1]
        return label_i, label_j

    dataset = get_dataset(infile)
    text_data = path.join(config.surv_dir, dataset, 'TextData')
    create_if_nonexist(text_data)

    post_image = {}
    fin = open(infile)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        post = fields[POST_INDEX]
        image = fields[IMAGE_INDEX]
        post_image[post] = image
    fin.close()

    rawtags_file = path.join(text_data, 'id.userid.rawtags.txt')
    fout = open(rawtags_file, 'w')
    fin = open(config.rawtag_file)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        post = fields[0]
        if post not in post_image:
            continue
        post = fields[POST_INDEX]
        image = post_image[post]
        user = fields[USER_INDEX]
        old_labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
        new_labels = []
        for old_label in old_labels:
            old_label = urllib.parse.unquote(old_label)
            old_label = old_label.lower()
            new_label = ''
            for c in old_label:
                if not c.isalnum():
                    continue
                new_label += c
            new_labels.append(new_label)
        labels = ' '.join(new_labels)
        fout.write('{}\t{}\t{}\n'.format(image, user, labels))
    fin.close()
    fout.close()

    lemmtags_file = path.join(text_data, 'id.userid.lemmtags.txt')
    fout = open(lemmtags_file, 'w')
    fin = open(rawtags_file)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        old_labels = fields[-1].split(' ')
        new_labels = []
        for old_label in old_labels:
            new_label = lemmatizer.lemmatize(old_label)
            new_labels.append(new_label)
        fields[-1] = ' '.join(new_labels)
        fout.write('{}\n'.format(FIELD_SEPERATOR.join(fields)))
    fin.close()
    fout.close()

    fin = open(lemmtags_file)
    label_users, label_images = {}, {}
    label_set = set()
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image, user = fields[0], fields[1]
        labels = fields[2].split()
        for label in labels:
            if label not in label_users:
                label_users[label] = set()
            label_users[label].add(user)
            if label not in label_images:
                label_images[label] = set()
            label_images[label].add(image)
            label_set.add(label)
    fin.close()
    tagfreq_file = path.join(text_data, 'lemmtag.userfreq.imagefreq.txt')
    fout = open(tagfreq_file, 'w')
    label_count = {}
    for label in label_set:
        label_count[label] = len(label_users[label]) # + len(label_images[label])
    sorted_label_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
    for label, _ in sorted_label_count:
        userfreq = len(label_users[label])
        imagefreq = len(label_images[label])
        fout.write('{} {} {}\n'.format(label, userfreq, imagefreq))
    fout.close()

    jointfreq_file = path.join(text_data, 'ucij.uuij.icij.iuij.txt')
    min_count = 4
    if not infile.endswith('.valid'):
        min_count = 8
    label_count = {}
    fin = open(lemmtags_file)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image, user = fields[0], fields[1]
        labels = fields[2].split()
        for label in labels:
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
    fin.close()
    jointfreq_icij_init = {}
    fin = open(lemmtags_file)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image, user = fields[0], fields[1]
        labels = fields[2].split()
        num_label = len(labels)
        for i in range(num_label - 1):
            for j in range(i + 1, num_label):
                label_i = labels[i]
                label_j = labels[j]
                if label_i == label_j:
                    continue
                if label_count[label_i] < min_count:
                    continue
                if label_count[label_j] < min_count:
                    continue
                key = _get_key(label_i, label_j)
                if key not in jointfreq_icij_init:
                    jointfreq_icij_init[key] = 0
                jointfreq_icij_init[key] += 1
    fin.close()
    keys = set()
    icij_images = {}
    iuij_images = {}
    fin = open(lemmtags_file)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image, user = fields[0], fields[1]
        labels = fields[2].split()
        num_label = len(labels)
        for i in range(num_label - 1):
            for j in range(i + 1, num_label):
                label_i = labels[i]
                label_j = labels[j]
                if label_i == label_j:
                    continue
                if label_i not in iuij_images:
                    iuij_images[label_i] = set()
                iuij_images[label_i].add(image)
                if label_j not in iuij_images:
                    iuij_images[label_j] = set()
                iuij_images[label_j].add(image)
                if label_count[label_i] < min_count:
                    continue
                if label_count[label_j] < min_count:
                    continue
                key = _get_key(label_i, label_j)
                if jointfreq_icij_init[key] < min_count:
                    continue
                keys.add(key)
                if key not in icij_images:
                    icij_images[key] = set()
                icij_images[key].add(image)
    fin.close()
    jointfreq_icij, jointfreq_iuij = {}, {}
    keys = sorted(keys)
    for key in keys:
        jointfreq_icij[key] = len(icij_images[key])
        label_i, label_j = _get_labels(key)
        label_i_images = iuij_images[label_i]
        label_j_images = iuij_images[label_j]
        jointfreq_iuij[key] = len(label_i_images.union(label_j_images))
    fout = open(jointfreq_file, 'w')
    for key in sorted(keys):
        label_i, label_j = _get_labels(key)
        fout.write('{} {} {} {} {} {}\n'.format(label_i, label_j, jointfreq_icij[key], jointfreq_iuij[key], jointfreq_icij[key], jointfreq_iuij[key]))
    fout.close()

    fin = open(lemmtags_file)
    vocab = set()
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image, user = fields[0], fields[1]
        labels = fields[2].split()
        for label in labels:
            if wordnet.synsets(label):
                vocab.add(label)
            else:
                pass
    fin.close()
    vocab_file = path.join(text_data, 'wn.%s.txt' % dataset)
    fout = open(vocab_file, 'w')
    for label in sorted(vocab):
        fout.write('{}\n'.format(label))
    fout.close()

def survey_feature_sets(infile):
    dataset = get_dataset(infile)
    image_sets = path.join(config.surv_dir, dataset, 'ImageSets')
    create_if_nonexist(image_sets)

    fout = open(path.join(image_sets, '%s.txt' % dataset), 'w')
    fin = open(infile)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image = fields[IMAGE_INDEX]
        fout.write('{}\n'.format(image))
    fin.close()
    fout.close()

    fout = open(path.join(image_sets, 'holdout.txt'), 'w')
    fout.close()

def survey_annotations(infile):
    dataset = get_dataset(infile)
    annotations = path.join(config.surv_dir, dataset, 'Annotations')
    create_if_nonexist(annotations)
    concepts = 'concepts.txt'
    
    label_set = set()
    label_images = {}
    image_set = set()
    fin = open(infile)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image = fields[IMAGE_INDEX]
        labels = fields[LABEL_INDEX].split()
        for label in labels:
            label_set.add(label)
            if label not in label_images:
                label_images[label] = []
            label_images[label].append(image)
        image_set.add(image)
    fin.close()
    fout = open(path.join(annotations, concepts), 'w')
    for label in sorted(label_set):
        fout.write('{}\n'.format(label))
    fout.close()

    concepts_dir = path.join(annotations, 'Image', concepts)
    create_if_nonexist(concepts_dir)
    image_list = sorted(image_set)
    for label in label_set:
        label_filepath = path.join(concepts_dir, '%s.txt' % label)
        fout = open(label_filepath, 'w')
        for image in image_list:
            assessment = -1
            if image in label_images[label]:
                assessment = 1
            fout.write('{} {}\n'.format(image, assessment))
        fout.close()

if __name__ == '__main__':
    create_if_nonexist(config.yfcc10k_dir)
    check_num_field()
    if not skip_if_exist(config.label_file):
        print('select top labels')
        select_top_label()
    if not skip_if_exist(config.raw_file):
        print('select posts')
        select_posts()
    if not skip_if_exist(config.data_file):
        print('tokenize dataset')
        tokenize_dataset()
        count_dataset()
    if (not skip_if_exist(config.train_file) or 
                not skip_if_exist(config.valid_file or 
                not skip_if_exist(config.vocab_file))):
        print('split dataset')
        split_dataset()
    if path.isdir(config.image_dir):
        print('collect images')
        # find ImageData/ -type f | wc -l
        collect_image(config.data_file, config.image_data_dir)
    # exit()
    if (not skip_if_exist(config.train_tfrecord or
                not skip_if_exist(config.valid_tfrecord))):
        print('create tfrecord')
        create_tfrecord(config.train_file)
        create_tfrecord(config.valid_file)
    check_tfrecord(config.train_tfrecord, True)
    check_tfrecord(config.valid_tfrecord, False)
    return
    
    print('create survey data')
    survey_image_data(config.train_file)
    survey_image_data(config.valid_file)
    survey_text_data(config.train_file)
    survey_text_data(config.valid_file)
    survey_feature_sets(config.train_file)
    survey_feature_sets(config.valid_file)
    survey_annotations(config.train_file)
    survey_annotations(config.valid_file)