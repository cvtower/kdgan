kdgan_dir=$HOME/Projects/kdgan/kdgan
checkpoint_dir=$kdgan_dir/checkpoints


for tch_keep_prob in 0.95 0.90 0.85 0.80 0.75
do
  for tch_weight_decay in 0.01 0.001 0.0001 0.00001
  do
    python pretrain_tch.py \
      --checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_tch \
      --save_path=$checkpoint_dir/mdlcompr_mnist_tch/model \
      --dataset_dir=$HOME/Projects/data/mnist \
      --tch_keep_prob=$tch_keep_prob \
      --num_epoch=200
    # target=0.9932
    # bstacc=0.9951
    # exit
  done
done


python train_gan.py \
  --dis_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_dis \
  --gen_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --dataset_dir=$HOME/Projects/data/mnist
exit


python pretrain_gen.py \
  --checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --save_path=$checkpoint_dir/mdlcompr_mnist_gen/model \
  --dataset_dir=$HOME/Projects/data/mnist \
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


python chk_model.py \
  --dataset_dir=$HOME/Projects/data/mnist \
  --model_name=lenet \
  --preprocessing_name='lenet'
exit


