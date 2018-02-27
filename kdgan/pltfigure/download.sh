server=xiaojie@10.100.228.149 # cz
server=xiaojie@10.100.228.181 # xw

pickle_dir=Projects/kdgan_xw/kdgan/pickles
src_yfcc_dir=/home/xiaojie/${pickle_dir}
dst_yfcc_dir=$HOME/${pickle_dir}

scp $server:${src_yfcc_dir}/* ${dst_yfcc_dir}