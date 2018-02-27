kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pretrained_dir=${checkpoint_dir}/pretrained
pickle_dir=${kdgan_dir}/pickles

num_epoch=200
gen_model_ckpt=${checkpoint_dir}/gen_vgg_16.ckpt
dis_model_ckpt=${checkpoint_dir}/dis_vgg_16.ckpt
tch_model_ckpt=${checkpoint_dir}/tagrecom_yfcc10k_tch.ckpt

gen_model_p=${pickle_dir}/tagrecom_yfcc10k_gen@${num_epoch}.p
tch_model_p=${pickle_dir}/tagrecom_yfcc10k_tch@${num_epoch}.p

dataset=yfcc10k
image_model=vgg_16

python pretrain_gen.py \
  --gen_model_ckpt=${gen_model_ckpt} \
  --gen_model_p=${gen_model_p} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --learning_rate=0.05 \
  --num_epoch=${num_epoch}
exit

python pretrain_dis.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --learning_rate=0.05 \
  --num_epoch=${num_epoch}
exit

python pretrain_tch.py \
  --tch_model_ckpt=${tch_model_ckpt} \
  --tch_model_p=${tch_model_p} \
  --dataset=$dataset \
  --image_model=${image_model} \
  --text_weight_decay=0.0 \
  --tch_learning_rate=0.01 \
  --num_epoch=${num_epoch}
exit

python train_kdgan.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --image_model=vgg_16 \
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

python train_gan.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
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

python train_kd.py \
  --dataset=yfcc10k \
  --model_name=vgg_16 \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --tch_model_ckpt=$checkpoint_dir/tch.ckpt \
  --feature_size=4096 \
  --kd_lamda=0.9999 \
  --temperature=10.0 \
  --num_epoch=200
exit




