kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
pretrained_dir=${checkpoint_dir}/pretrained
pickle_dir=${kdgan_dir}/pickles
picture_dir=${kdgan_dir}/pictures

num_epoch=200
gen_model_ckpt=${checkpoint_dir}/gen_vgg_16.ckpt
dis_model_ckpt=${checkpoint_dir}/dis_vgg_16.ckpt
tch_model_ckpt=${checkpoint_dir}/tagrecom_yfcc10k_tch.ckpt

gen_model_p=${pickle_dir}/tagrecom_yfcc10k_gen@${num_epoch}.p
tch_model_p=${pickle_dir}/tagrecom_yfcc10k_tch@${num_epoch}.p

python plot.py \
  --gen_model_p=${gen_model_p} \
  --tch_model_p=${tch_model_p} \
  --epsfile=${picture_dir}/gen_vs_tch.eps
