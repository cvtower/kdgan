from os import path

from collections import Counter
import numpy as np
import tensorflow as tf


tf.flags.DEFINE_boolean('dev', False, 'dev')
tf.flags.DEFINE_string('facebook_infile', None, '')
tf.flags.DEFINE_string('ngrams', None, '--ngrams=2,3,4,5')

tf.flags.DEFINE_string('train_tfrecord', None, '')
tf.flags.DEFINE_string('valid_tfrecord', None, '')
tf.flags.DEFINE_string('label_file', None, '')
tf.flags.DEFINE_string('vocab_file', None, '')

tf.flags.DEFINE_integer('batch_size', 128, '')
tf.flags.DEFINE_integer('train_steps', 5000, '')
tf.flags.DEFINE_integer('valid_steps', 100, '')
tf.flags.DEFINE_integer('num_epochs', 100, '')


tf.flags.DEFINE_integer('num_oov_vocab_buckets', 20,
        'number of hash buckets to use for OOV words')
tf.flags.DEFINE_integer('embedding_dimension', 10,
        'dimension of word embedding')
tf.flags.DEFINE_boolean('use_ngrams', False,
        'use character ngrams in embedding')
tf.flags.DEFINE_integer('num_ngram_buckets', 1000000,
        'number of hash buckets for ngrams')
tf.flags.DEFINE_integer('ngram_embedding_dimension', 10,
        'dimension of word embedding')

tf.flags.DEFINE_integer('num_threads', 1,
        'number of reader threads')

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)
if FLAGS.dev:
    tf.logging.set_verbosity(tf.logging.DEBUG)

TEXT_KEY = 'text'
LABELS_KEY = 'labels'
NGRAMS_KEY = 'ngrams'

def parse_ngrams(ngrams):
    ngrams = [int(g) for g in ngrams.split(',')]
    ngrams = [g for g in ngrams if (g > 1 and g < 7)]
    return ngrams

def generate_ngrams(words, ngrams):
    nglist = []
    for ng in ngrams:
        for word in words:
            nglist.extend([word[n:n+ng] for n in range(len(word)-ng+1)])
    return nglist

def load_label_to_id():
    fin = open(FLAGS.label_file)
    labels = [label.strip() for label in fin.readlines()]
    fin.close()
    label_to_id = dict(zip(labels, range(len(labels))))
    return label_to_id

def load_id_to_label():
    label_to_id = load_label_to_id()
    id_to_label = dict([(v, k) for k, v in label_to_id.items()])
    return id_to_label

def parse_facebook_infile(infile, ngrams):
    label_prefix = '__label__'
    examples = []
    for line in open(infile):
        words = line.split()
        labels = []
        for word in words:
            if word.startswith(label_prefix):
                label = word[len(label_prefix):]
                labels.append(label)
            else:
                break
        text = words[len(labels):]
        if len(labels) == 0:
            print('no labels')
            exit()
        if len(text) == 0:
            print('no text')
            exit()
        example = {LABELS_KEY: labels, TEXT_KEY:text}
        if ngrams:
            example[NGRAMS_KEY] = generate_ngrams(text, ngrams)
        examples.append(example)
    return examples

def build_tfrecord(example, label_to_id):
    text = example[TEXT_KEY]
    labels = example[LABELS_KEY]
    ngrams = example.get(NGRAMS_KEY, None)
    record = tf.train.Example()
    text = [tf.compat.as_bytes(x) for x in text]
    record.features.feature[TEXT_KEY].bytes_list.value.extend(text)
    
    # labels = [tf.compat.as_bytes(x) for x in labels]
    # record.features.feature[LABELS_KEY].bytes_list.value.extend(labels)
    label_ids = [label_to_id[label] for label in labels]
    labels = np.zeros((len(label_to_id),), dtype=np.int64)
    labels[label_ids] = 1
    record.features.feature[LABELS_KEY].int64_list.value.extend(labels)

    if ngrams is not None:
        ngrams = [tf.compat.as_bytes(x) for x in ngrams]
        record.features.feature[NGRAMS_KEY].bytes_list.value.extend(ngrams)
    return record

def write_examples(examples, tfrecord_file):
    label_to_id = load_label_to_id()
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    for n, example in enumerate(examples):
        record = build_tfrecord(example, label_to_id)
        writer.write(record.SerializeToString())

def write_vocab(examples, vocab_file, label_file):
    words = Counter()
    labels = set()
    for example in examples:
        words.update(example[TEXT_KEY])
        labels.update(example[LABELS_KEY])
    with open(vocab_file, 'w') as fout:
        for word in words.most_common():
            fout.write(word[0] + '\n')
    with open(label_file, 'w') as fout:
        labels = sorted(list(labels))
        for label in labels:
            fout.write(str(label) + '\n')

def cleanse():
    if not FLAGS.facebook_infile:
        print('no --facebook_infile')
        exit()
    if not FLAGS.label_file:
        print('no --label_file')
        exit()
    ngrams = None
    if FLAGS.ngrams:
        ngrams = parse_ngrams(FLAGS.ngrams)
    examples = parse_facebook_infile(FLAGS.facebook_infile, ngrams)
    tfrecord_file = path.join(FLAGS.facebook_infile + '.tfrecord')
    write_examples(examples, tfrecord_file)
    vocab_file = path.join(FLAGS.facebook_infile + '.vocab')
    label_file = path.join(FLAGS.facebook_infile + '.label')
    write_vocab(examples, vocab_file, label_file)

def get_parse_spec(use_ngrams, num_label):
    parse_spec = {
        TEXT_KEY:tf.VarLenFeature(dtype=tf.string),
        LABELS_KEY:tf.FixedLenFeature([num_label], tf.int64, default_value=tf.zeros([num_label], dtype=tf.int64)),
    }
    if use_ngrams:
        parse_spec[NGRAMS_KEY] = tf.VarLenFeature(dtype=tf.string)
    return parse_spec

def train():
    vocab_size = len(open(FLAGS.vocab_file).readlines())
    id_to_label = load_id_to_label()
    num_label = len(id_to_label)
    print('#vocab={} #label={}'.format(vocab_size, num_label))

    parse_spec = get_parse_spec(FLAGS.use_ngrams, num_label)
    features = tf.contrib.learn.read_batch_features(
            FLAGS.train_tfrecord,
            FLAGS.batch_size,
            parse_spec,
            tf.TFRecordReader,
            num_epochs=FLAGS.num_epochs,
            reader_num_threads=FLAGS.num_threads)
    features['text'] = tf.sparse_tensor_to_dense(features['text'], default_value=' ')
    # text_ts = features.get(TEXT_KEY)
    # label_ts = features.get(LABELS_KEY)
    # text_ts = features[TEXT_KEY]
    # label_ts = features[LABELS_KEY]
    from tensorflow.python.framework import errors
    from tensorflow.python.ops import variables
    from tensorflow.python.training import coordinator
    from tensorflow.python.training import queue_runner_impl
    with tf.Session() as session:
      session.run(variables.local_variables_initializer())
      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(session, coord=coord)

      try:
        while not coord.should_stop():
            feature_np, = session.run([features])
            # res = session.run([text_ts, label_ts])
            text_np, label_np = feature_np['text'], feature_np['labels']
            print(type(text_np), text_np.shape, type(label_np), label_np.shape)
            for i in range(FLAGS.batch_size):
                label_ids = [j for j in range(num_label) if label_np[i,j] != 0]
                labels = [id_to_label[label_id] for label_id in label_ids]
                text = [text_np[i,j].decode('utf-8') for j in range(text_np.shape[1]) if text_np[i,j] != b' ']
                text = ' '.join(text)
                print(str(text), labels)
                input()
            input()
      except errors.OutOfRangeError:
        pass
      finally:
        coord.request_stop()

      coord.join(threads)

def main(_):
    # cleanse()
    train()

if __name__ == '__main__':
    tf.app.run()

