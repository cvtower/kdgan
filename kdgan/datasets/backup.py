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



def create_annotations(infile):

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