kdgan_dir=$HOME/Projects/kdgan/kdgan
checkpoint_dir=$kdgan_dir/checkpoints



python pretrain_gen.py \
  --checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --save_path=$checkpoint_dir/mdlcompr_mnist_gen/model \
  --dataset_dir=$HOME/Projects/data/mnist \
  --preprocessing_name='lenet' \
  --num_epoch=200
# target=0.9854
# bstacc=0.9862 # no dropout no l2
# bstacc=0.9884 # wt dropout wt l2
exit


python train_kd.py \
  --gen_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --tch_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --preprocessing_name='lenet'
exit


python pretrain_tch.py \
  --checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_tch \
  --save_path=$checkpoint_dir/mdlcompr_mnist_tch/model \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet \
  --preprocessing_name='lenet'
  --num_epoch=200
# target=0.9932
# bstacc=0.9951
exit


python chk_model.py \
  --dataset_dir=$HOME/Projects/data/mnist \
  --model_name=lenet \
  --preprocessing_name='lenet'
exit


