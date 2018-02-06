kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=$kdgan_dir/checkpoints

train_size=500

python train_gan.py \
  --dis_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_dis \
  --gen_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet_v1 \
  --dis_keep_prob=0.75 \
  --dis_weight_decay=0.00001 \
  --optimizer=adam \
  --dis_learning_rate=0.001 \
  --dis_learning_rate_decay_factor=0.96 \
  --gen_learning_rate=0.001 \
  --gen_learning_rate_decay_factor=0.98 \
  --learning_rate_decay_type=exponential \
  --train_size=$train_size \
  --num_epoch=100 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10
# bstacc=0.7839 # 500 train examples
# bstacc=0.9548 # 5000 train examples
exit


python train_kd.py \
  --kd_hard_pct=1.0 \
  --temperature=1.0 \
  --gen_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --tch_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_tch \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet_v1 \
  --tch_keep_prob=0.75 \
  --optimizer=adam \
  --tch_opt_epsilon=1e-8 \
  --gen_learning_rate=0.001 \
  --gen_learning_rate_decay_factor=0.98 \
  --learning_rate_decay_type=exponential \
  --train_size=$train_size \
  --num_epoch=200
exit


python pretrain_dis.py \
  --dis_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_dis \
  --dis_save_path=$checkpoint_dir/mdlcompr_mnist_dis/model \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet_v1 \
  --dis_keep_prob=0.75 \
  --dis_weight_decay=0.00001 \
  --optimizer=adam \
  --dis_learning_rate=0.001 \
  --dis_learning_rate_decay_factor=0.96 \
  --learning_rate_decay_type=exponential \
  --train_size=$train_size \
  --num_epoch=200
# target=0.9932
# bstacc=0.7679 # 500 train examples
# bstacc=0.9756 # 5000 train examples
exit


python pretrain_tch.py \
  --tch_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_tch \
  --tch_save_path=$checkpoint_dir/mdlcompr_mnist_tch/model \
  --dataset_dir=$HOME/Projects/data/mnist \
  --tch_model_name=lenet_v1 \
  --tch_keep_prob=0.75 \
  --tch_weight_decay=0.00001 \
  --optimizer=adam \
  --tch_opt_epsilon=1e-8 \
  --tch_learning_rate=0.001 \
  --tch_learning_rate_decay_factor=0.96 \
  --train_size=$train_size \
  --num_epoch=200
# target=0.9932
# bstacc=0.9951
# avg=2.62s/epoch
exit


python pretrain_gen.py \
  --gen_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_gen \
  --gen_save_path=$checkpoint_dir/mdlcompr_mnist_gen/model \
  --dataset_dir=$HOME/Projects/data/mnist \
  --optimizer=adam \
  --gen_learning_rate=0.001 \
  --gen_learning_rate_decay_factor=0.98 \
  --learning_rate_decay_type=exponential \
  --train_size=$train_size \
  --num_epoch=200
# target=0.9854
# bstacc=0.9862 # no dropout no l2
# bstacc=0.9884 # wt dropout wt l2
# avg=0.89s/epoch
# bstacc=0.7554 # 500 train examples
# bstacc=0.7653 # 500 train examples
# bstacc=0.9511 # 5000 train examples
exit


for tch_keep_prob in 0.95 0.90 0.85 0.80 0.75
do
  for tch_weight_decay in 0.0001 0.00005 0.00001
  do
    python pretrain_tch.py \
      --tch_checkpoint_dir=$checkpoint_dir/mdlcompr_mnist_tch \
      --tch_save_path=$checkpoint_dir/mdlcompr_mnist_tch/model \
      --dataset_dir=$HOME/Projects/data/mnist \
      --tch_keep_prob=$tch_keep_prob \
      --tch_weight_decay=$tch_weight_decay \
      --optimizer=adam \
      --tch_learning_rate=0.001 \
      --tch_learning_rate_decay_factor=0.96 \
      --num_epoch=200
    # target=0.9932
    # bstacc=0.9951
  done
done
exit


python chk_model.py \
  --dataset_dir=$HOME/Projects/data/mnist \
  --model_name=lenet
exit


