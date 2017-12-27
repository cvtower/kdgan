# kdgan

ssh xiaojiew1@10.100.229.246
ssh xiaojie@10.100.228.181

conda create -n py27 python=2.7 # create virtualenv for baseline
conda create -n py34 python=3.4 # create virtualenv for kdgan


# jingwei/util/simpleknn/lib/linux/libsearch.so: cannot open shared object file: No such file or directory
sudo apt-get install libboost-dev
./jingwei/util/simpleknn/build.sh