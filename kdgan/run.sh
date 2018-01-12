kdgan_dir=$HOME/Projects/kdgan/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
pretrained_dir=$checkpoint_dir/pretrained

python train_gan.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --dis_model_ckpt=$checkpoint_dir/dis_vgg_16.ckpt \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --feature_size=4096 \
  --dis_weight_decay=0.0 \
  --gen_weight_decay=0.0 \
  --num_epoch=100 \
  --num_dis_epoch=50 \
  --num_gen_epoch=10
exit

python pretrain_dis.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --dis_model_ckpt=$checkpoint_dir/dis_vgg_16.ckpt \
  --feature_size=4096 \
  --learning_rate=0.05 \
  --num_epoch=200
# 373s best hit=0.7690
exit

python pretrain_gen.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --feature_size=4096 \
  --learning_rate=0.05 \
  --num_epoch=200
# 386s best hit=0.7707
exit

python pretrain_tch.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --tch_model_ckpt=$checkpoint_dir/tch.ckpt \
  --tch_weight_decay=0.0 \
  --learning_rate=0.01 \
  --num_epoch=200
# 0232s best hit=0.9657
exit

python train_kdgan.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --dis_model_ckpt=$checkpoint_dir/dis_vgg_16.ckpt \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --tch_model_ckpt=$checkpoint_dir/tch.ckpt \
  --feature_size=4096 \
  --kd_lamda=0.00001 \
  --temperature=10.0 \
  --dis_weight_decay=0.0 \
  --gen_weight_decay=0.0 \
  --tch_weight_decay=0.0 \
  --num_epoch=100 \
  --num_dis_epoch=1 \
  --num_gen_epoch=1 \
  --num_tch_epoch=1
exit
