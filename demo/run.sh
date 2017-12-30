DATADIR=/Users/xiaojiew1/Projects/data/yfcc100m/fbinput
DATASET=yfcc10k

TRAIN_FBINPUT=$DATADIR/${DATASET}.train
VALID_FBINPUT=$DATADIR/${DATASET}.valid
LABEL_FILE=$DATADIR/${DATASET}.label

# python main.py \
#     --facebook_infile=${TRAIN_FBINPUT} \
#     --label_file=$LABEL_FILE \
#     --ngrams=2,3,4

# python main.py \
#     --facebook_infile=${VALID_FBINPUT} \
#     --label_file=$LABEL_FILE \
#     --ngrams=2,3,4


TRAIN_TFRECORD=$TRAIN_FBINPUT.tfrecord
VOCAB_FILE=$TRAIN_FBINPUT.vocab
VALID_TFRECORD=$VALID_FBINPUT.tfrecord
EXPORT_DIR=$DATADIR/models/${DATASET}
OUTPUT=$DATADIR/models/${DATASET}

python main.py \
    --train_tfrecord=$TRAIN_TFRECORD \
    --valid_tfrecord=$VALID_TFRECORD \
    --label_file=$LABEL_FILE \
    --vocab_file=$VOCAB_FILE \
    --model_dir=$OUTPUT \
    --export_dir=$EXPORT_DIR \
    --learning_rate=0.01 \
    --nolog_device_placement \
    --fast