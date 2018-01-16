kdgan_dir=$HOME/Projects/kdgan/kdgan
checkpoint_dir=$kdgan_dir/checkpoints


python pretrain_gen.py \
  --checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --save_path=$checkpoint_dir/mdlcompr_mnist_gen/model \
  --dataset_dir=$HOME/Projects/data/mnist \
  --preprocessing_name='lenet' \
  --dropout_keep_prob=0.80 \
  --weight_decay=0.01 \
  --batch_size=128 \
  --num_epoch=200
exit

python train_kd.py \
  --gen_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --tch_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --preprocessing_name='lenet'
exit


python chk_model.py \
  --dataset_dir=$HOME/Projects/data/mnist \
  --model_name=lenet \
  --preprocessing_name='lenet'
exit

python pretrain_tch.py \
  --checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_tch \
  --save_path=$checkpoint_dir/mdlcompr_mnist_tch/model \
  --dataset_dir=$HOME/Projects/data/mnist \
  --model_name=lenet \
  --preprocessing_name='lenet'
exit

