import config

from datasets import imagenet

import operator
import os
import random
import shutil

from os import path

################################################################
#
# prepare yfcc10k
#
################################################################

SPACE_PLACEHOLDER = '+'

FIELD_SEPERATOR = '\t'
LABEL_SEPERATOR = ','

POST_INDEX = 0
USER_INDEX = 1
IMAGE_INDEX = 2
LABEL_INDEX = -1
NUM_FIELD = 6

TOT_LABEL = 100
TOT_POST = 10000
MIN_USER_CORE = 10
MAX_USER_CORE = 100
MIN_LABEL_CORE = 100

def select_labels(infile):
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

    fin = open(infile)
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

    label_set, num_label, = set(), 0
    for label, _ in label_count:
        if num_label == TOT_LABEL:
            break
        if label in invalid:
            continue
        if label not in valid:
            continue
        label_set.add(label)
        num_label += 1
    return label_set

def in_label_set(labels, label_set):
    old_labels = labels.split(LABEL_SEPERATOR)
    new_labels = []
    for label in old_labels:
        if label not in label_set:
            continue
        new_labels.append(label)
    if len(new_labels) == 0:
        return False
    return True

def cleanse(labels, label_set):
    old_labels = labels.split(LABEL_SEPERATOR)
    new_labels = []
    for label in old_labels:
        if label not in label_set:
            continue
        new_labels.append(label)
    return new_labels

def get_image_path(image_dir, image_url):
    fields = image_url.split('/')
    image_path = path.join(image_dir, fields[-2], fields[-1])
    return image_path

def create_if_nonexist(outdir):
    if not path.exists(outdir):
        os.makedirs(outdir)

def count_post(user_posts):
    tot_post = 0
    for user, posts in user_posts.items():
        tot_post += len(posts)
    return tot_post

def personalize(infile, label_set):
    copy = False
    if path.isdir(config.image_dir):
        copy = True

    user_posts = {}
    fin = open(infile)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        user, labels = fields[USER_INDEX], fields[LABEL_INDEX]
        image_url = fields[IMAGE_INDEX]
        if not in_label_set(labels, label_set):
            continue
        labels = cleanse(labels, label_set)
        fields[LABEL_INDEX] = LABEL_SEPERATOR.join(labels)
        fields[IMAGE_INDEX] = path.basename(image_url)
        if user not in user_posts:
            user_posts[user] = []
        user_posts[user].append(FIELD_SEPERATOR.join(fields))
        if copy:
            src_filepath = get_image_path(config.image_dir, image_url)
            filename = path.basename(image_url)
            user_dir = path.join(config.image_dir, user)
            create_if_nonexist(user_dir)
            dst_filepath = path.join(user_dir, filename)
            shutil.copyfile(src_filepath, dst_filepath)
    fin.close()

    user_posts_cpy = user_posts
    user_posts = {}
    for user in user_posts_cpy.keys():
        posts = user_posts_cpy[user]
        num_post = len(posts)
        if num_post < MIN_USER_CORE:
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

    user_posts_cpy = user_posts
    user_posts = {}
    for user in user_posts_cpy.keys():
        posts = user_posts_cpy[user]
        num_post = len(posts)
        num_post = min(num_post // 5 * 5, MAX_USER_CORE)
        not_keep = len(posts) - num_post
        user_posts[user] = []
        count = 0
        for post in posts:
            keep = False
            fields = post.split(FIELD_SEPERATOR)
            labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
            for label in labels:
                if (label_count[label] - 1) < MIN_LABEL_CORE:
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
        if (num_post // 5) != 0:
            num_post = num_post // 5 * 5
            user_posts[user] = posts[:num_post]
            for post in posts[num_post:]:
                fields = post.split(FIELD_SEPERATOR)
                labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
                for label in labels:
                    label_count[label] -= 1

    tot_post = count_post(user_posts)
    dif_post = tot_post - TOT_POST
    print('{} to be removed'.format(dif_post))
    users = list(user_posts.keys())
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
            if (label_count[label] - user_label_count[label]) < MIN_LABEL_CORE:
                keep = True
                break
        if not keep:
            num_post = len(posts)
            if dif_post - num_post < 0:
                if num_post - dif_post < MIN_USER_CORE:
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

    tot_post = count_post(user_posts)
    user_count = {}
    for user, posts in user_posts.items():
        user_count[user] = len(posts)
    dif_post = tot_post - TOT_POST
    print('{} to be removed'.format(dif_post))
    sorted_user_count = sorted(user_count.items(), key=operator.itemgetter(1), reverse=True)
    user_posts_cpy = user_posts
    user_posts = {}
    for user, posts in user_posts_cpy.items():
        keep_posts = []
        disc_posts = []
        for post in posts:
            keep = False
            fields = post.split(FIELD_SEPERATOR)
            labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
            for label in labels:
                if (label_count[label] - 1) < MIN_LABEL_CORE:
                    keep = True
                    break
            if keep:
                keep_posts.append(post)
            else:
                disc_posts.append(post)
                for label in labels:
                    label_count[label] -= 1

        num_keep = max(len(keep_posts), MIN_USER_CORE)
        if num_keep % 5 != 0:
            num_keep = (num_keep // 5 + 1) * 5

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

    fout = open(config.data_filepath, 'w')
    for user in user_posts.keys():
        posts = user_posts[user]
        for post in posts:
            fout.write('{}\n'.format(post))
    fout.close()

def save_data(user_posts, outfile):
    fout = open(outfile, 'w')
    for user, posts in user_posts.items():
        for post in posts:
            fout.write('{}\n'.format(post))
    fout.close()

def validate_data(infile, label_set):
    print(path.basename(infile), len(label_set))
    fin = open(infile)
    while True:
        line = fin.readline()
        if not line:
            break
        fields = line.strip().split(FIELD_SEPERATOR)
        if len(fields) != NUM_FIELD:
            print('line {} has no {} fields'.format(fields[0], NUM_FIELD))
            exit()
        labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
        for label in labels:
            label_set.discard(label)
    print(path.basename(infile), len(label_set))
    assert len(label_set) == 0

def split_data(label_set):
    split_seed = 66

    user_posts = {}
    fin = open(config.data_filepath)
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
    random.seed(split_seed)
    for user, posts in user_posts.items():
        random.shuffle(posts)
        num_post = len(posts)
        if (num_post % 5) == 0:
            seperator = int(num_post * 0.80)
        else:
            seperator = int(num_post * 0.80) + 1
        train_user_posts[user] = posts[:seperator]
        valid_user_posts[user] = posts[seperator:]

    save_data(train_user_posts, config.train_filepath)
    save_data(valid_user_posts, config.valid_filepath)

    validate_data(config.train_filepath, label_set.copy())
    validate_data(config.valid_filepath, label_set.copy())

def compute_statistics():
    create_if_nonexist(config.temp_dir)
    user_count = {}
    fin = open(config.data_filepath)
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
    fin = open(config.data_filepath)
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
    with open(outfile, 'w') as fout:
        for label, count in sorted_label_count:
            fout.write('{}\t{}\n'.format(label, count))

def prepare_data():
    label_set = select_labels(config.sample_filepath)
    personalize(config.sample_filepath, label_set)
    print('#label={}'.format(len(label_set)))
    split_data(label_set)

def baseline_data():
    survey_data = config.yfcc_dir
    image_data = path.join(survey_data, config.dataset, 'ImageData')
    create_if_nonexist(image_data)
    posts = set()
    fin = open(config.data_filepath)
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
    fin = open(config.sample_filepath)
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
        dst_filepath = path.join(image_data, filename)
        shutil.copyfile(src_filepath, dst_filepath)
    fin.close()
    print('#image={}'.format(len(images)))

if __name__ == '__main__':
    prepare_data()
    compute_statistics()

    baseline_data()


