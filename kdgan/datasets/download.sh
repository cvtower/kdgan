server=xiaojie@10.100.228.181
yfcc_dir=Projects/data/yfcc100m

survey_data=survey_data.zip
src_yfcc_dir=/home/xiaojie/$yfcc_dir
dst_yfcc_dir=$HOME/$yfcc_dir
[ -f $dst_yfcc_dir/$survey_data ] || scp $server:$src_yfcc_dir/$survey_data $dst_yfcc_dir

dataset=yfcc10k

src_data_dir=$server:/home/xiaojie/$yfcc_dir/$dataset
dst_data_dir=$HOME/$yfcc_dir/$dataset
echo $src_data_dir
echo $dst_data_dir

[ -d $dst_data_dir ] || mkdir $dst_data_dir
# scp $src_data_dir/$dataset.label $dst_data_dir
# scp $src_data_dir/$dataset.vocab $dst_data_dir
# scp $src_data_dir/$dataset.train $dst_data_dir
# scp $src_data_dir/$dataset.valid $dst_data_dir

src_precomputed_dir=$src_data_dir/Precomputed
dst_precomputed_dir=$dst_data_dir/Precomputed
[ -d ${dst_precomputed_dir} ] || mkdir ${dst_precomputed_dir}

model_name=vgg_16
scp ${src_precomputed_dir}/${dataset}_${model_name}_000.valid.tfrecord ${dst_precomputed_dir}
for i in $(seq -w 000 499)
do
  filename=${dataset}_${model_name}_${i}.train.tfrecord
  if [ -f ${dst_precomputed_dir}/$filename ]
  then
    continue
  fi
  scp ${src_precomputed_dir}/$filename ${dst_precomputed_dir}
done

model_name=inception_resnet_v2
scp ${src_precomputed_dir}/${dataset}_${model_name}_000.valid.tfrecord ${dst_precomputed_dir}
for i in $(seq -w 000 49)
do
  filename=${dataset}_${model_name}_${i}.train.tfrecord
  if [ -f ${dst_precomputed_dir}/$filename ]
  then
    continue
  fi
  scp ${src_precomputed_dir}/$filename ${dst_precomputed_dir}
done
