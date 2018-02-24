kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
pretrained_dir=$checkpoint_dir/pretrained

python pretrain_gen.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --image_model=vgg_16 \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --feature_size=4096 \
  --learning_rate=0.05 \
  --num_epoch=200
exit

