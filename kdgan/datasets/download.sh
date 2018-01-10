SERVER=xiaojie@10.100.228.181
dataset=yfcc10k
model_name=vgg_16

yfcc_dir=Projects/data/yfcc100m
src_data_dir=$SERVER:/home/xiaojie/$yfcc_dir/$dataset
dst_data_dir=$HOME/$yfcc_dir/$dataset
echo $src_data_dir
echo $dst_data_dir

# [ -d $dst_data_dir ] || mkdir $dst_data_dir
# scp $src_data_dir/$dataset.label $dst_data_dir
# scp $src_data_dir/$dataset.vocab $dst_data_dir
# scp $src_data_dir/$dataset.train $dst_data_dir
# scp $src_data_dir/$dataset.valid $dst_data_dir

src_precomputed_dir=$src_data_dir/Precomputed
dst_precomputed_dir=$dst_data_dir/Precomputed
[ -d $dst_precomputed_dir ] || mkdir $dst_precomputed_dir
scp $src_precomputed_dir/${dataset}_${model_name}_000.valid.tfrecord ${dst_precomputed_dir}
for i in $(seq -w 000 249)
do
  if [ $i -gt 99 ]
  then
    break
  fi
  scp $src_precomputed_dir/${dataset}_${model_name}_${i}.train.tfrecord ${dst_precomputed_dir}
done
