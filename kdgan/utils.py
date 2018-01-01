from kdgan import config

def save_collection(coll, outfile):
    with open(outfile, 'w') as fout:
        for elem in coll:
            fout.write('%s\n' % elem)

def load_collection(infile):
    with open(infile) as fin:
        coll = [elem.strip() for elem in fin.readlines()]
    return coll

def load_sth_to_id(infile):
    with open(infile) as fin:
        sth_list = [sth.strip() for sth in fin.readlines()]
    sth_to_id = dict(zip(sth_list, range(len(sth_list))))
    return sth_to_id

def load_label_to_id():
    label_to_id = load_sth_to_id(config.label_file)
    return label_to_id

def load_token_to_id():
    vocab_to_id = load_sth_to_id(config.vocab_file)
    return vocab_to_id

def load_id_to_sth(infile):
    with open(infile) as fin:
        sth_list = [sth.strip() for sth in fin.readlines()]
    id_to_sth = dict(zip(range(len(sth_list)), sth_list))
    return id_to_sth

def load_id_to_label():
    id_to_label = load_id_to_sth(config.label_file)
    return id_to_label

def load_id_to_token():
    id_to_vocab = load_id_to_sth(config.vocab_file)
    return id_to_vocab