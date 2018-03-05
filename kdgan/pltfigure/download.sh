cy_server=xiaojie@10.100.228.151 # cy
cz_server=xiaojie@10.100.228.149 # cz
xw_server=xiaojie@10.100.228.181 # xw

pickle_dir=Projects/kdgan_xw/kdgan/pickles
src_yfcc_dir=/home/xiaojie/${pickle_dir}
dst_yfcc_dir=$HOME/${pickle_dir}
[ -d $dst_yfcc_dir ] || mkdir -p $dst_yfcc_dir

################################################################
#
# convergence rate
#
################################################################

num_epoch=200
# scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_gen@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_tch@${num_epoch}.p ${dst_yfcc_dir}
# scp ${cz_server}:${src_yfcc_dir}/tagrecom_yfcc10k_gan@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/tagrecom_yfcc10k_kdgan@${num_epoch}.p ${dst_yfcc_dir}

train_size=50
# scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_gen@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_tch@${num_epoch}.p ${dst_yfcc_dir}
# scp ${cz_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_gan@${num_epoch}.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist${train_size}_kdgan@${num_epoch}.p ${dst_yfcc_dir}

################################################################
#
# parameter tuning
#
################################################################

# scp ${cz_server}:${src_yfcc_dir}/mdlcompr_mnist*_kdgan_*.p ${dst_yfcc_dir}
# scp ${xw_server}:${src_yfcc_dir}/mdlcompr_mnist*_kdgan_*.p ${dst_yfcc_dir}

download() {
  train_size=$1
  for alpha in 0.0 0.2 0.4 0.6 0.8 1.0
  do
    for beta in 0.0 0.2 0.4 0.6 0.8 1.0
    do
      for gamma in 0.0 0.2 0.4 0.6 0.8 1.0
      do
        epk_learning_curve_p=${pickle_dir}/mdlcompr_mnist${train_size}_kdgan_${alpha}_${beta}_${gamma}.p
        scp ${cy_server}:${epk_learning_curve_p} ../pickles/
      done
    done
  done
}

train_size=50
download ${train_size}





