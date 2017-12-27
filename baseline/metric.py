import config
from os import path

FIELD_SEPERATOR = '\t'
IMAGE_INDEX = 2
LABEL_INDEX = -1
LABEL_SEPERATOR = ','

def main():
    result_filepath = path.join(config.logs_dir, 'tagvote.res')
    valid_filepath = config.valid_filepath
    cutoff = 3

    image_labels = {}
    fin = open(valid_filepath)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image = fields[IMAGE_INDEX]
        labels = fields[LABEL_INDEX].split(LABEL_SEPERATOR)
        image_labels[image] = labels
    fin.close()

    results = {}
    fin = open(result_filepath)
    while True:
        line = fin.readline().strip()
        if not line:
            break
        fields = line.split(FIELD_SEPERATOR)
        image = fields[0]
        results[image] = []
        for i in range(1,len(fields)):
            label = fields[i].split(':')[0]
            results[image].append(label)
    fin.close()

    rec_list = []
    for image in image_labels.keys():
        count = 0
        for i in range(cutoff):
            if results[image][i] in image_labels[image]:
                count += 1
        rec = 1.0 * count / len(image_labels[image])
        rec_list.append(rec)
    rec = sum(rec_list) / len(rec_list)
    print('#{0}\t{1:.4f}'.format(len(rec_list), rec))

if __name__ == '__main__':
    main()