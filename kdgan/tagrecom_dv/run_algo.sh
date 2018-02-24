kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
pretrained_dir=$checkpoint_dir/pretrained

gen_model_ckpt=${checkpoint_dir}/gen_vgg_16.ckpt
dis_model_ckpt=${checkpoint_dir}/dis_vgg_16.ckpt

python train_gan.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --gen_model_ckpt=${gen_model_ckpt} \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --image_model=vgg_16 \
  --feature_size=4096 \
  --dis_weight_decay=0.0 \
  --gen_weight_decay=0.0 \
  --learning_rate=0.05 \
  --num_epoch=500 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10
exit

python pretrain_dis.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --dis_model_ckpt=${dis_model_ckpt} \
  --feature_size=4096 \
  --learning_rate=0.05 \
  --num_epoch=100
exit

python pretrain_gen.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --image_model=vgg_16 \
  --gen_model_ckpt=${gen_model_ckpt} \
  --feature_size=4096 \
  --learning_rate=0.05 \
  --num_epoch=200
exit

