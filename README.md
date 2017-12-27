# kdgan

ssh xiaojiew1@10.100.229.246
ssh xiaojie@10.100.228.181

conda create -n py27 python=2.7 # create virtualenv for baseline
conda create -n py34 python=3.4 # create virtualenv for kdgan


# do_tagprop.sh
sudo apt-get install libboost-dev
./jingwei/util/simpleknn/build.sh
./do_getknn.sh yfcc8k yfcc8k vgg-verydeep-16fc7relu 0 1 1
./do_getknn.sh yfcc8k yfcc2k vgg-verydeep-16fc7relu 0 1 1

conda install libgcc