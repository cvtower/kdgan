kdgan_dir=$HOME/Projects/kdgan_xw/kdgan
checkpoint_dir=$kdgan_dir/checkpoints
pickle_dir=${kdgan_dir}/pickles

cz_server=xiaojie@10.100.228.149 # cz
xw_server=xiaojie@10.100.228.181 # xw

dis_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_dis
gen_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_gen
tch_model_ckpt=${checkpoint_dir}/mdlcompr_mnist${train_size}_tch

# scp ${cz_server}:${checkpoint_dir}/mdlcompr_mnist${train_size}* ${checkpoint_dir}

train_size=50
batch_size=5
for intelltch_weight in 0.1 0.3 0.5 0.7 0.9
do
  for distilled_weight in 0.1 0.5 1.0 5.0 10.0
  do

epk_learning_curve_p=${pickle_dir}/mdlcompr_mnist${train_size}_kdgan_${intelltch_weight}_${distilled_weight}.p
python train_kdgan.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --epk_learning_curve_p=${epk_learning_curve_p} \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --dis_learning_rate=1e-3 \
  --gen_learning_rate=5e-4 \
  --tch_learning_rate=5e-4 \
  --num_epoch=100 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10 \
  --kdgan_model=odgan \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=mimic \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1 \
  --intelltch_weight=${intelltch_weight} \
  --distilled_weight=${distilled_weight}

  done
done


train_size=500
batch_size=10
for intelltch_weight in 0.1 0.3 0.5 0.7 0.9
do
  for distilled_weight in 0.1 0.5 1.0 5.0 10.0
  do

epk_learning_curve_p=${pickle_dir}/mdlcompr_mnist${train_size}_kdgan_${intelltch_weight}_${distilled_weight}.p
python train_kdgan.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --epk_learning_curve_p=${epk_learning_curve_p} \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --dis_learning_rate=1e-3 \
  --gen_learning_rate=5e-4 \
  --tch_learning_rate=5e-4 \
  --num_epoch=100 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10 \
  --kdgan_model=odgan \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=mimic \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1 \
  --intelltch_weight=${intelltch_weight} \
  --distilled_weight=${distilled_weight}

  done
done


train_size=5000
batch_size=50
for intelltch_weight in 0.1 0.3 0.5 0.7 0.9
do
  for distilled_weight in 0.1 0.5 1.0 5.0 10.0
  do

epk_learning_curve_p=${pickle_dir}/mdlcompr_mnist${train_size}_kdgan_${intelltch_weight}_${distilled_weight}.p
python train_kdgan.py \
  --dis_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_dis \
  --gen_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_gen \
  --tch_model_ckpt=$checkpoint_dir/mdlcompr_mnist${train_size}_tch \
  --epk_learning_curve_p=${epk_learning_curve_p} \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dis_model_name=lenet \
  --gen_model_name=mlp \
  --tch_model_name=lenet \
  --optimizer=adam \
  --train_size=$train_size \
  --batch_size=$batch_size \
  --dis_learning_rate=1e-3 \
  --gen_learning_rate=5e-4 \
  --tch_learning_rate=5e-4 \
  --num_epoch=100 \
  --num_dis_epoch=20 \
  --num_gen_epoch=10 \
  --num_tch_epoch=10 \
  --kdgan_model=odgan \
  --num_negative=20 \
  --num_positive=5 \
  --kd_model=mimic \
  --noisy_ratio=0.1 \
  --noisy_sigma=0.1 \
  --intelltch_weight=${intelltch_weight} \
  --distilled_weight=${distilled_weight}

  done
done
