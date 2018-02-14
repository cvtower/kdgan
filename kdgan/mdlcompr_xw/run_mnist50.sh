kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
train_size=50
batch_size=5

# scp xiaojie@10.100.228.149:${checkpoint_dir}/mdlcompr_mnist${train_size}* ${checkpoint_dir}

# mac
# scp xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_gen.data-00000-of-00001 ${checkpoint_dir}
# scp xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_gen.index ${checkpoint_dir}
# scp xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_gen.meta ${checkpoint_dir}
# scp xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_dis.data-00000-of-00001 ${checkpoint_dir}
# scp xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_dis.index ${checkpoint_dir}
# scp xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints/mdlcompr_mnist${train_size}_dis.meta ${checkpoint_dir}

# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_gen.data-00000-of-00001 xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_gen.index xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_gen.meta xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_dis.data-00000-of-00001 xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_dis.index xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints
# scp ${checkpoint_dir}/mdlcompr_mnist${train_size}_dis.meta xiaojie@10.100.228.149:/home/xiaojie/Projects/kdgan_xw/kdgan/checkpoints


python train_gan.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
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


python pretrain_gen.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=50 bstacc=0.5209 et=5s
exit


python pretrain_tch.py \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=50 bstacc=0.6343 et=21s
exit


python pretrain_dis.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200
#mnist=50 bstacc=0.6294 et=20s
exit


python train_kd.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --kd_model=mimic
#mnist=50 mimic=0.5690 iniacc=0.5209 et=6s
exit


python train_kd.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --kd_model=distn \
  --kd_soft_pct=0.7 \
  --temperature=3.0
#mnist=50 distn=0.5645 iniacc=0.5209 et=6s
exit


python train_kd.py \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --num_epoch=200 \
  --kd_model=noisy \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1
#mnist=50 noisy=0.5718 iniacc=0.5209 et=6s
exit


python train_kdgan.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
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
  --kdgan_model=ow \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=noisy \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1
#mnist=1000 kdgan_ow=0.8985 et=1459s
exit


python train_kdgan.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
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
  --kdgan_model=tw \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=mimic \
  --kd_soft_pct=0.3 \
  --temperature=3.0
#mnist=10000 kdgan_ow=0.9786 et=10419s
exit











