kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=${kdgan_dir}/checkpoints
train_size=50000
batch_size=128

dataset_dir=$HOME/Projects/data/cifar
train_filepath=${dataset_dir}/cifar-10-batches-bin/data_batch*
valid_filepath=${dataset_dir}/cifar-10-batches-bin/test_batch*

dis_model_ckpt=${checkpoint_dir}/mdlcompr_cifar${train_size}_dis.ckpt
std_model_ckpt=${checkpoint_dir}/mdlcompr_cifar${train_size}_std.ckpt
tch_model_ckpt=${checkpoint_dir}/mdlcompr_cifar${train_size}_tch.ckpt
tch_ckpt_dir=${checkpoint_dir}/mdlcompr_cifar_tch

python train_gan.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --std_model_ckpt=${std_model_ckpt} \
  --train_filepath=${dataset_dir}/cifar-10-batches-bin/data_batch* \
  --valid_filepath=${dataset_dir}/cifar-10-batches-bin/test_batch* \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --num_epoch=200 \
  --num_dis_epoch=20 \
  --num_std_epoch=10 \
  --num_negative=20 \
  --num_positive=5
exit


python pretrain_dis.py \
  --dis_model_ckpt=${dis_model_ckpt} \
  --train_filepath=${dataset_dir}/cifar-10-batches-bin/data_batch* \
  --valid_filepath=${dataset_dir}/cifar-10-batches-bin/test_batch* \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --learning_rate_decay_factor=0.96 \
  --num_epochs_per_decay=10.0 \
  --num_epoch=200
#cifar=50000 final=0.8402
exit


# run this command several times to get good results
python train_kd.py \
  --std_model_ckpt=${std_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --train_filepath=${train_filepath} \
  --valid_filepath=${valid_filepath} \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --optimizer=sgd \
  --std_learning_rate=0.01 \
  --num_epoch=200 \
  --kd_model=mimic \
  --kd_soft_pct=0.1
#cifar=50000 final=0.8304
exit


python train_kd.py \
  --std_model_ckpt=${std_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --train_filepath=${train_filepath} \
  --valid_filepath=${valid_filepath} \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --num_epoch=200 \
  --kd_model=distn \
  --kd_soft_pct=0.1
  --temperature=3.0
#cifar=50000 final=0.8366
exit


python train_kd.py \
  --std_model_ckpt=${std_model_ckpt} \
  --tch_model_ckpt=${tch_model_ckpt} \
  --train_filepath=${train_filepath} \
  --valid_filepath=${valid_filepath} \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --num_epoch=200 \
  --kd_model=noisy \
  --kd_soft_pct=0.1 \
  --noisy_ratio=0.01 \
  --noisy_sigma=0.01
#cifar=50000 final=0.8282
exit


python pretrain_tch.py \
  --tch_model_ckpt=${tch_model_ckpt} \
  --tch_ckpt_dir=${tch_ckpt_dir} \
  --train_filepath=${dataset_dir}/cifar-10-batches-bin/data_batch* \
  --valid_filepath=${dataset_dir}/cifar-10-batches-bin/test_batch* \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --learning_rate_decay_factor=0.95 \
  --num_epochs_per_decay=10.0 \
  --num_epoch=200
#cifar=50000 final=0.8836 #tn_batch=78125
exit


python pretrain_std.py \
  --std_model_ckpt=${std_model_ckpt} \
  --train_filepath=${dataset_dir}/cifar-10-batches-bin/data_batch* \
  --valid_filepath=${dataset_dir}/cifar-10-batches-bin/test_batch* \
  --train_size=${train_size} \
  --batch_size=${batch_size} \
  --learning_rate_decay_factor=0.96 \
  --num_epochs_per_decay=10.0 \
  --num_epoch=200
#cifar=50000 final=0.8402
exit


checkpoint_dir=$HOME/Projects/kdgan_xw/kdgan/checkpoints
tch_ckpt_dir=${checkpoint_dir}/mdlcompr_cifar_tch
rm -rf ${tch_ckpt_dir}
scp -r xiaojie@10.100.228.181:${tch_ckpt_dir} ${checkpoint_dir}


checkpoint_dir=$HOME/Projects/kdgan_xw/kdgan/checkpoints
train_size=50000
std_model_ckpt=${checkpoint_dir}/mdlcompr_cifar${train_size}_std
scp xiaojie@10.100.228.181:${std_model_ckpt}* ${checkpoint_dir}





