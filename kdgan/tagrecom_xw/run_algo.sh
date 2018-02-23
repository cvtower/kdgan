kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pretrained_dir=${checkpoint_dir}/pretrained

variant=basic
dataset=yfcc10k
image_model=vgg_16
dis_model_ckpt=${checkpoint_dir}/dis_$variant.ckpt
gen_model_ckpt=${checkpoint_dir}/gen_$variant.ckpt
tch_model_ckpt=${checkpoint_dir}/tch_$variant.ckpt

python train_kd.py \
  --gen_model_ckpt=${gen_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=fix \
  --gen_learning_rate=0.1 \
  --kd_model=distn \
  --kd_soft_pct=0.1 \
  --temperature=3.0 \
  --epk_train=1.0 \
  --epk_valid=0.0 \
  --num_epoch=200
exit

python pretrain_tch.py \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=fix \
  --gen_learning_rate=0.1 \
  --epk_train=0.8 \
  --epk_valid=0.2 \
  --num_epoch=200
# bsthit=0.8340 et=731s
exit

python pretrain_dis.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=fix \
  --gen_learning_rate=0.1 \
  --epk_train=0.8 \
  --epk_valid=0.2 \
  --num_epoch=200
# bsthit=0.8240 et=726s
exit

python pretrain_gen.py \
  --gen_model_ckpt=${gen_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=fix \
  --gen_learning_rate=0.1 \
  --num_epoch=200
exit

python train_kdgan.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --gen_model_ckpt=${gen_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10
exit



python train_gan.py \
  --dataset=yfcc10k \
  --image_model=vgg_16 \
  --dis_model_ckpt=$checkpoint_dir/dis_vgg_16.ckpt \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --gan_figure_data=$figure_data_dir/gan_vgg_16.csv \
  --feature_size=4096 \
  --dis_weight_decay=0.0 \
  --gen_weight_decay=0.0 \
  --learning_rate=0.05 \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10
# best hit=0.7817
exit



