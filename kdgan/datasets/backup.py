import config

from datasets import imagenet

import operator
import os
import random
import shutil
import urllib

from os import path

from nltk.corpus import wordnet as wn
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


TOT_LABEL = 100
TOT_POST = 10000
MIN_USER_CORE = 10
MAX_USER_CORE = 100
MIN_LABEL_CORE = 100

TRAIN_RATIO = 0.90
UNIT_POST = 10
TRAIN_RATIO = 0.80
UNIT_POST = 5

COPY_IMAGE = True

################################################################
#
# create baseline data (must create kdgan data first)
#
################################################################

def count_datasize(infile):
    with open(infile) as fout:
        for i, l in enumerate(fout):
            pass
    datasize = i + 1
    return datasize

def get_dataset(datasize):
    dataset = 'yfcc{}k'.format(datasize // 1000)
    return dataset

def collect_images(infile):
    datasize = count_datasize(infile)
    dataset = get_dataset(datasize)
    image_data = path.join(config.surv_dir, dataset, 'ImageData')
    create_if_nonexist(image_data)

    posts = set()
    fin = open(infile)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        post = fields[POST_INDEX]
        posts.add(post)
    fin.close()
    print('#post={}'.format(len(posts)))

    images = set()
    fout = open(path.join(image_data, '%s.txt' % dataset), 'w')
    fin = open(config.init_sample_filepath)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        post = fields[0]
        if post not in posts:
            continue
        image_url = fields[IMAGE_INDEX]
        src_filepath = get_image_path(config.image_dir, image_url)
        filename = path.basename(image_url)
        image = filename.split('_')[0]
        images.add(image)
        filename = '%s.jpg' % image
        fout.write('{}\n'.format(filename))
        dst_filepath = path.join(image_data, filename)
        if not COPY_IMAGE:
            continue
        if not path.isfile(src_filepath):
            continue
        shutil.copyfile(src_filepath, dst_filepath)
    fin.close()
    fout.close()
    print('#image={}'.format(len(images)))

def create_text_data(infile):
    datasize = count_datasize(infile)
    dataset = get_dataset(datasize)
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
    print('#post={}'.format(len(post_image)))

    rawtags_filename = 'id.userid.rawtags.txt'
    rawtags_filepath = path.join(text_data, rawtags_filename)
    fout = open(rawtags_filepath, 'w')
    fin = open(config.init_sample_filepath)
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

    lemmtags_filename = 'id.userid.lemmtags.txt'
    lemmtags_filepath = path.join(text_data, lemmtags_filename)
    fout = open(lemmtags_filepath, 'w')
    fin = open(rawtags_filepath)
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

    fin = open(lemmtags_filepath)
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
    tagfreq_filename = 'lemmtag.userfreq.imagefreq.txt'
    tagfreq_filepath = path.join(text_data, tagfreq_filename)
    fout = open(tagfreq_filepath, 'w')

    label_count = {}
    for label in label_set:
        label_count[label] = len(label_users[label]) # + len(label_images[label])
    sorted_label_count = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
    
    for label, _ in sorted_label_count:
        userfreq = len(label_users[label])
        imagefreq = len(label_images[label])
        fout.write('{} {} {}\n'.format(label, userfreq, imagefreq))
    fout.close()

    jointfreq_filename = 'ucij.uuij.icij.iuij.txt'
    jointfreq_filepath = path.join(text_data, jointfreq_filename)
    seperator = '###'
    def getkey(label_i, label_j):
        if label_i < label_j:
            key = label_i + seperator + label_j
        else:
            key = label_j + seperator + label_i
        return key
    def getpair(key):
        fields = key.split(seperator)
        label_i, label_j = fields[0], fields[1]
        return label_i, label_j

    if not infile.endswith('.valid'):
        min_count = 6
    else:
        min_count = 3
    label_count = {}
    fin = open(lemmtags_filepath)
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
    print('#label={}'.format(len(label_count)))
    pair_icij_temp = {}
    fin = open(lemmtags_filepath)
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
                key = getkey(label_i, label_j)
                if key not in pair_icij_temp:
                    pair_icij_temp[key] = 0
                pair_icij_temp[key] += 1
    fin.close()
    pair_icij_images = {}
    pair_iuij_images = {}
    fin = open(lemmtags_filepath)
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

                if label_i not in pair_iuij_images:
                    pair_iuij_images[label_i] = set()
                pair_iuij_images[label_i].add(image)
                if label_j not in pair_iuij_images:
                    pair_iuij_images[label_j] = set()
                pair_iuij_images[label_j].add(image)

                if label_count[label_i] < min_count:
                    continue
                if label_count[label_j] < min_count:
                    continue

                key = getkey(label_i, label_j)
                if pair_icij_temp[key] < min_count:
                    continue

                if key not in pair_icij_images:
                    pair_icij_images[key] = set()
                pair_icij_images[key].add(image)
    fin.close()
    pair_icij, pair_iuij = {}, {}
    keys = set()
    for key, iuij_images in pair_icij_images.items():
        pair_icij[key] = len(iuij_images)
        label_i, label_j = getpair(key)
        label_i_images = pair_iuij_images[label_i]
        label_j_images = pair_iuij_images[label_j]
        label_ij_images = label_i_images.union(label_j_images)
        pair_iuij[key] = len(label_ij_images)
        keys.add(key)
    fout = open(jointfreq_filepath, 'w')
    for key in sorted(keys):
        label_i, label_j = getpair(key)
        fout.write('{} {} {} {} {} {}\n'.format(label_i, label_j,
                pair_icij[key], pair_iuij[key], pair_icij[key], pair_iuij[key]))
    fout.close()

    fin = open(lemmtags_filepath)
    vocab = set()
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image, user = fields[0], fields[1]
        labels = fields[2].split()
        for label in labels:
            if wn.synsets(label):
                vocab.add(label)
            else:
                pass
    fin.close()
    vocab_filename = 'wn.%s.txt' % dataset
    vocab_filepath = path.join(text_data, vocab_filename)
    fout = open(vocab_filepath, 'w')
    for label in sorted(vocab):
        fout.write('{}\n'.format(label))
    fout.close()

def create_feature_sets(infile):
    datasize = count_datasize(infile)
    dataset = get_dataset(datasize)
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

def create_annotations(infile):
    datasize = count_datasize(infile)
    dataset = get_dataset(datasize)
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
        labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
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

    # if not infile.endswith('.valid'):
    #     return

    concepts_dir = path.join(annotations, 'Image', concepts)
    create_if_nonexist(concepts_dir)
    image_list = sorted(image_set)
    for label in label_set:
        label_filepath = path.join(concepts_dir, '%s.txt' % label)
        fout = open(label_filepath, 'w')
        for image in image_list:
            rele = -1
            if image in label_images[label]:
                rele = 1
            fout.write('{} {}\n'.format(image, rele))
        fout.close()

def create_baseline_data():
    # post = '99951995'
    # yfcc100m_filepath = '/data/yfcc100m/yfcc100m_dataset'
    # fin = open(yfcc100m_filepath)
    # label_count = {}
    # while True:
    #     line = fin.readline().strip()
    #     if not line:
    #         break
    #     fields = line.split(FIELD_SEPERATOR)
    #     if fields[0] != post:
    #         continue
    #     for field in fields:
    #         print(field)
    #     break
    # fin.close()

    # collect_images(config.train_filepath)
    create_text_data(config.train_filepath)
    # create_feature_sets(config.train_filepath)
    # create_annotations(config.train_filepath)

    # collect_images(config.valid_filepath)
    create_text_data(config.valid_filepath)
    # create_feature_sets(config.valid_filepath)
    # create_annotations(config.valid_filepath)

def select_lemmatizer():
    def read_tags(infile):
        fin = open(infile)
        tags = []
        while True:
            line = fin.readline().strip()
            if not line:
                break
            fields = line.split(FIELD_SEPERATOR)
            labels = fields[-1].split(' ')
            tags.extend(labels)
        fin.close()
        return tags

    text_data = path.join(config.data_dir, 'jingwei/train10k/TextData')
    rawtags_filepath = path.join(text_data, 'id.userid.rawtags.txt')
    lemmtags_filepath = path.join(text_data, 'id.userid.lemmtags.txt')
    rawtags = read_tags(rawtags_filepath)
    lemmtags = read_tags(lemmtags_filepath)
    for rawtag, lemmtag in zip(rawtags, lemmtags):
        mylemm = lemmatizer.lemmatize(rawtag)
        if mylemm != lemmtag:
            print('{}\t{}\t{}'.format(rawtag, lemmtag, mylemm))
            input()
        else:
            # print(rawtag, lemmtag, mylemm)
            pass

def main():
    # user_set = set()
    # fin = open(infile)
    # while True:
    #     line = fin.readline().strip()
    #     if not line:
    #         break
    #     fields = line.split(FIELD_SEPERATOR)
    #     image, user = fields[0], fields[1]
    #     labels = fields[2].split()
    #     user_set.add(user)
    # fin.close()
    # print('#user={}'.format(len(user_set)))

    def preprocess(text):
        return text

    def postprocess(text):
        # import re
        # tag_re = re.compile(r'<.*?>')
        # text = tag_re.sub('', text)

        return text

    def tokenize(text):
        from nltk import word_tokenize
        tokens = word_tokenize(text)

        def in_synsets(token):
            if wn.synsets(token):
                return True
            else:
                return False

        from nltk.corpus import stopwords
        stopwords = set(stopwords.words('english'))
        
        from nltk.stem import PorterStemmer
        # stemmer = PorterStemmer()
        from nltk.stem.snowball import SnowballStemmer
        stemmer = SnowballStemmer('english')

        def _tokenize(tokens):
            tokens = [token for token in tokens if in_synsets(token)]
            tokens = [token for token in tokens if not token in stopwords]
            tokens = [stemmer.stem(token) for token in tokens]
            return tokens

        tokens = _tokenize(tokens)

        if len(tokens) == 0:
            from nltk.tokenize import RegexpTokenizer
            tokenizer = RegexpTokenizer('[a-z]+')
            tokens = tokenizer.tokenize(text)
            tokens = _tokenize(tokens)

        return tokens

    # text = 'building-harsh buliding a bulding'
    # tokens = tokenize(text)
    # print(tokens)
    # exit()

    infile = config.train_filepath
    infile = config.valid_filepath
    outdir = '/Users/xiaojiew1/Projects/kdgan/temp/yfcc10k'
    fin = open(infile)
    outfile = path.join(outdir, path.basename(infile))
    fout = open(outfile, 'w')
    
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        post = fields[POST_INDEX]
        labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
        title = fields[TITLE_INDEX]
        description = fields[DESCRIPTION_INDEX]
        # print('{0}\n{0}'.format('#'*80))
        title = preprocess(title)
        # print(title)
        # print('#'*80)
        # print()
        description = preprocess(description)
        # print(description)
        # print('#'*80)
        title = postprocess(title)
        # print(title)
        # print('#'*80)
        # print()
        description = postprocess(description)
        # print(description)
        # print('{0}\n{0}'.format('#'*80))
        text = ' '.join([title, description])
        tokens = tokenize(text)
        # print('{2}\n{0}\n{2}{1}\n{2}'.format(text, tokens, '#'*80))
        # input()
        for label in labels:
            fout.write('__label__%s ' % label)
        fout.write('%s\n' % ' '.join(tokens))
        # if len(tokens) == 0:
        #     print('text: {}'.format(text))
        #     print('labels: {}'.format(labels))
    fout.close()
    fin.close()

if __name__ == '__main__':
    create_kdgan_data()
    # summarize_data()

    # select_lemmatizer()
    # create_baseline_data()
    # matlab -nodisplay -nosplash -nodesktop -r "run('extract_vggnet.m');"

    # main()