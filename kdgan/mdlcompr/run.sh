kdgan_dir=$HOME/Projects/kdgan/kdgan
checkpoint_dir=$kdgan_dir/checkpoints

python pretrain_gen.py \
  --checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --save_path=$checkpoint_dir/mdlcompr_mnist_gen/model \
  --dataset_dir=$HOME/Projects/data/mnist \
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

