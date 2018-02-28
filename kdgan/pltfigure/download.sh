cz_server=xiaojie@10.100.228.149 # cz
xw_server=xiaojie@10.100.228.181 # xw

pickle_dir=Projects/kdgan_xw/kdgan/pickles
src_yfcc_dir=/home/xiaojie/${pickle_dir}
dst_yfcc_dir=$HOME/${pickle_dir}
[ -d $dst_yfcc_dir ] || mkdir -p $dst_yfcc_dir

num_epoch=200
# scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_gen@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_tch@${num_epoch}.p ${dst_yfcc_dir}
# scp ${cz_server}:${src_yfcc_dir}/tagrecom_yfcc10k_gan@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_kdgan@${num_epoch}.p ${dst_yfcc_dir}

train_size=50
scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_gen@${num_epoch}.p ${dst_yfcc_dir}
scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_tch@${num_epoch}.p ${dst_yfcc_dir}
scp ${cz_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_gan@${num_epoch}.p ${dst_yfcc_dir}
scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_kdgan@${num_epoch}.p ${dst_yfcc_dir}
