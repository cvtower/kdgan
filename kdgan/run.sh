kdgan_dir=$HOME/Projects/kdgan/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
pretrained_dir=$checkpoint_dir/pretrained

python train_kdgan.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --dis_model_ckpt=$checkpoint_dir/dis_vgg_16.ckpt \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --tch_model_ckpt=$checkpoint_dir/tch.ckpt \
  --feature_size=4096 \
  --beta=0.00001 \
  --temperature=10.0 \
  --dis_weight_decay=0.0 \
  --gen_weight_decay=0.0 \
  --tch_weight_decay=0.0 \
  --num_epoch=100 \
  --num_dis_epoch=1 \
  --num_gen_epoch=1 \
  --num_tch_epoch=1
exit
