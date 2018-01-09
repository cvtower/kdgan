kdgan_dir=$HOME/Projects/kdgan/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
pretrained_dir=$checkpoint_dir/pretrained

python pretrain_gen.py \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --model_name=vgg_16 \
  --feature_size=4096 \
  --num_epoch=200
exit

python pretrain_dis.py \
  --dis_model_ckpt=$checkpoint_dir/dis_vgg_16.ckpt \
  --model_name=vgg_16 \
  --feature_size=4096 \
  --num_epoch=200
exit
 # 469s best hit=0.6574

python train_gan.py \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --model_name=vgg_16 \
  --feature_size=4096 \
  --tch_weight_decay=0.0 \
  --gen_weight_decay=0.0 \
  --num_epoch=1 \
  --num_dis_epoch=50 \
  --num_gen_epoch=50
exit

python train_kd.py \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --tch_model_ckpt=$checkpoint_dir/tch.ckpt \
  --model_name=vgg_16 \
  --feature_size=4096 \
  --beta=0.00001 \
  --temperature=10.0 \
  --num_epoch=200
exit

python pretrain_tch.py \
  --tch_model=tch \
  --model_name=vgg_16 \
  --tch_weight_decay=0.0 \
  --num_epoch=200
exit
# 339s best hit=0.9443
