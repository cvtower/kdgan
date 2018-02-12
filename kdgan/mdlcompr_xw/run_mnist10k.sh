kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
train_size=10000
batch_size=100

python pretrain_dis.py \
  --dis_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=50 bstacc=0.6536
exit


python pretrain_tch.py \
  --tch_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=10000 bstacc=0.9899 et=116s
exit


python pretrain_gen.py \
  --gen_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=10000 bstacc=0.9699 et=38s
exit


python train_kdgan.py \
  --dis_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10 \
  --num_negative=20 \
  --num_positive=2 \
  --kd_model=distn \
  --kd_soft_pct=0.7 \
  --temperature=3.0
exit


python train_gan.py \
  --dis_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_gen_epoch=2 \
  --num_negative=20 \
  --num_positive=5
exit


python train_kd.py \
  --gen_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_ckpt_file=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --kd_model=noisy \
  --kd_soft_pct=0.7 \
  --temperature=3.0
exit






