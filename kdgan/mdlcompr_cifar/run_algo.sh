kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
train_size=50
batch_size=5

python pretrain_std.py \
  --std_model_ckpt=${checkpoint_dir}/mdlcompr_cifar${train_size}_std \
  --dataset_dir=$HOME/Projects/data/cifar \
  --std_model_name=lenet \
  --optimizer=adam \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --num_epoch=20
#cifar=50 bstacc=0.6536
exit