kdgan_dir=/home/xiaojie/Projects/kdgan/kdgan
pretrained_dir=$kdgan_dir/checkpoints/pretrained


python pretrain_tch.py \
  --embedding_size=5 \
  --num_epochs=1000

exit

python pretrain_gen.py \
  --model_name=vgg_16 \
  --feature_size=4096 \
  --num_epochs=100

# exit