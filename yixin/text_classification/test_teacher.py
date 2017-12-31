import tensorflow as tf
from teacher_model import Teacher 
import sys
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
from random import *

sent_file_name = '../../data/yfcc10k.train'
#tag_file_name = '../../data/dev_tags'
sent_file_name_test = '../../data/yfcc10k.valid'
pre_trained_embedding_file = '../../data/wordVec/word_embedding.pkl'

def handle_sent_data(sent_file):
    sent_list = []
    tag_list = []
    max_length = 0
    with open(sent_file, encoding='latin-1') as f:
        for line in f:
            sent = line.split('\t')[4].split('+')
            if max_length < len(sent):
                max_length = len(sent)
            sent_list.append(sent)
            tag_list.append(line.split('\t')[5].strip().split(',')) 

    return sent_list, tag_list, max_length

    #return sent_list

def preprocess(sent_list):
    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    processed_list = []
    for sent in sent_list:
        new_list = []
        for word in sent:
            if word not in stopwords and word.isalpha():
                new_list.append(lemmatizer.lemmatize(word))
        processed_list.append(new_list)

    return processed_list

def get_seqLen(sent_list):
    seqLen = []
    for sent in sent_list:
        seqLen.append(len(sent)+1)
    
    return seqLen

def create_look_up_table(text, initial_index):
    dic_v2i = {}
    dic_i2v = {}
    if initial_index != 0:
        dic_v2i['unk'] = 0
        dic_i2v[0] = 'unk'
    index = initial_index
    for sent in text:
        for word in sent:
            try:
                test = dic_v2i[word]
            except:
                dic_v2i[word] = index
                index += 1

    for key in dic_v2i:
        dic_i2v[dic_v2i[key]] = key
    return dic_v2i, dic_i2v


def gen_sents(sent_list, id_dict, max_length):
    sent_matrix = []
    for sent in sent_list:
        sent_id_list = [0 for i in range(max_length)]
        for i in range(len(sent)):
            word = sent[i]
            try:
                _id = id_dict[word] 
            except:
                _id = 0
            sent_id_list[i] = _id
    
        sent_matrix.append(sent_id_list)
    
    return sent_matrix

def gen_tag(tag_list, tag_dict):
    tag_len = len(tag_dict)
    tag_matrix = [[0 for i in range(tag_len)] for j in range(len(tag_list))]
    for i in range(len(tag_list)):
        tags = tag_list[i]
        for tag in tags:
            tag_matrix[i][tag_dict[tag]] = 1
    
    return tag_matrix

def get_batch(data, tag, seqLen, index, batch_size):
    return data[index:index+batch_size], tag[index:index+batch_size], seqLen[index:index+batch_size], index + batch_size

def evaluate(logits, labels, cutoff, normalize):
    predictions = np.argsort(-logits, axis=1)[:,:cutoff]
    batch_size, _ = labels.shape
    scores = []
    for batch in range(batch_size):
        label_bt = labels[batch,:]
        label_bt = np.nonzero(label_bt)[0]
        prediction_bt = predictions[batch,:]
        num_label = len(label_bt)
        present = 0
        for label in label_bt:
            if label in prediction_bt:
                present += 1
        score = present
        if score > 0:
            score *= (1.0 / normalize(cutoff, num_label))
        # print('score={0:.4f}'.format(score))
        scores.append(score)
    score = np.mean(scores)
    return score

def precision(logits, labels, cutoff):
    def normalize(cutoff, num_label):
        return min(cutoff, num_label)
    prec = evaluate(logits, labels, cutoff, normalize)
    # print('prec={0:.4f}'.format(prec))
    return prec

def recall(logits, labels, cutoff):
    def normalize(cutoff, num_label):
        return num_label
    rec = evaluate(logits, labels, cutoff, normalize)
    # print('rec={0:.4f}'.format(rec))
    return rec

def create_pretrained_embeddings(id2word_dict):
    print('loading dictionary...')
    word_emb_dict = pickle.load(open(pre_trained_embedding_file, 'rb'))
    print('fetch word embeddings...')
    word_emb = []
    for key in id2word_dict:
        word = id2word_dict[key]
        if word not in word_emb_dict:
            emb = [2*(random()-0.5) for i in range(300)]
            print('not in:', word)
        else:
            emb = word_emb_dict[word]
        word_emb.append(emb)
    
    return tf.constant(word_emb, dtype=tf.float32)



def main():
    sent_list,tag_list, max_length = handle_sent_data(sent_file_name)
    sent_list_test,tag_list_test, max_length_test = handle_sent_data(sent_file_name_test) 
    print("training sentences:", len(sent_list), "max length:", max_length)
    print("test sentences:", len(sent_list_test), "max length:", max_length_test)
    max_length = max(max_length, max_length_test)
    cleaned_sent_list = preprocess(sent_list)
    cleaned_sent_list_test = preprocess(sent_list_test)
    seqLen = get_seqLen(cleaned_sent_list)
    seqLen_test = get_seqLen(cleaned_sent_list_test) 
    assert len(seqLen) == len(cleaned_sent_list)

    # create look up table for sent and tag
    vocab2id, id2vocab = create_look_up_table(cleaned_sent_list, 1)
    tag2id, id2tag = create_look_up_table(tag_list,0)

    #word_embed = create_pretrained_embeddings(id2vocab)
    word_embed = pickle.load(open('word_emb.pkl', 'rb'))
    word_embed_tensor = tf.constant(word_embed, dtype=tf.float32)
    #print(word_embed)
    #exit()

    total_sents = gen_sents(cleaned_sent_list, vocab2id, max_length)
    total_true_tag = gen_tag(tag_list, tag2id)


    sents_test = gen_sents(cleaned_sent_list_test, vocab2id, max_length)
    true_tag_test = gen_tag(tag_list_test, tag2id) 

    #print(tag_list) 
    #exit()

    wordNum = len(vocab2id)
    tagNum = len(tag2id)
    emb_dim = 300
    sample_num = 4
    lamda = 0.1
    cutoff = 2
    #epoch = 10
    batch_size = 128

    teacher = Teacher(wordNum, tagNum, emb_dim, sample_num, lamda, max_length,param=word_embed_tensor)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    index = 0
    for i in range(100000):
        #get next batch
        batch_sents, batch_tag, batch_seqLen, index = get_batch(total_sents, total_true_tag, seqLen, index, batch_size)
        if index > len(cleaned_sent_list):
            index = 0

        _, loss= sess.run([teacher.test_updates,teacher.test_loss],
                feed_dict={teacher.sents: batch_sents, teacher.true_tag: batch_tag, teacher.seqlen:batch_seqLen})
        
        #print(i)

        if i%100 == 0:
            test_loss, prob =sess.run([teacher.test_loss, teacher.tag_prob],
                feed_dict={teacher.sents: sents_test, teacher.true_tag: true_tag_test, teacher.seqlen:seqLen_test })

            
            print(i)
            print("test loss:", test_loss)
            print("test precision:", precision(prob, np.array(true_tag_test), cutoff))
        





if __name__ == '__main__':
    main()