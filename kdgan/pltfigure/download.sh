cz_server=xiaojie@10.100.228.149 # cz
xw_server=xiaojie@10.100.228.181 # xw

pickle_dir=Projects/kdgan_xw/kdgan/pickles
src_yfcc_dir=/home/xiaojie/${pickle_dir}
dst_yfcc_dir=$HOME/${pickle_dir}

[ -d $dst_yfcc_dir ] || mkdir -p $dst_yfcc_dir
scp ${cz_server}:${src_yfcc_dir}/tagrecom_yfcc10k_gan@200.p ${dst_yfcc_dir}
scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_kdgan@200.p ${dst_yfcc_dir}
scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_gen@200.p ${dst_yfcc_dir}
scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_tch@200.p ${dst_yfcc_dir}
