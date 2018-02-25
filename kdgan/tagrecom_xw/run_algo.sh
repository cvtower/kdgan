kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pretrained_dir=${checkpoint_dir}/pretrained

variant=basic
dataset=yfcc10k
image_model=vgg_16
dis_model_ckpt=${checkpoint_dir}/dis_$variant.ckpt
gen_model_ckpt=${checkpoint_dir}/gen_$variant.ckpt
tch_model_ckpt=${checkpoint_dir}/tch_$variant.ckpt

python pretrain_gen.py \
  --gen_model_ckpt=${gen_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --gen_learning_rate=0.05 \
  --num_epoch=200
exit

python pretrain_dis.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --dis_learning_rate=0.05 \
  --num_epoch=200
exit

python pretrain_tch.py \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --tch_learning_rate=0.05 \
  --epk_train=0.95 \
  --epk_valid=0.05 \
  --num_epoch=500
exit

python train_kd.py \
  --gen_model_ckpt=${gen_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --gen_learning_rate=0.05 \
  --kd_model=distn \
  --kd_soft_pct=0.3 \
  --temperature=3.0 \
  --num_epoch=200
exit

python train_kd.py \
  --gen_model_ckpt=${gen_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --gen_learning_rate=0.001 \
  --kd_model=mimic \
  --num_epoch=200
exit

python train_gan.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --gen_model_ckpt=${gen_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --image_weight_decay=0.0 \
  --optimizer=sgd \
  --learning_rate_decay_type=exp \
  --dis_learning_rate=0.05 \
  --num_epochs_per_decay=20.0 \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10
exit

python train_kdgan.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --dis_model_ckpt=$checkpoint_dir/dis_vgg_16.ckpt \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --tch_model_ckpt=$checkpoint_dir/tch.ckpt \
  --kdgan_figure_data=$figure_data_dir/kdgan_vgg_16.csv \
  --feature_size=4096 \
  --kd_lamda=0.9999 \
  --temperature=10.0 \
  --dis_weight_decay=0.0 \
  --gen_weight_decay=0.0 \
  --tch_weight_decay=0.0 \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10
# 12517s best hit=0.7973
exit



