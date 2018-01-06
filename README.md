# kdgan
pip install -Ue .

remote.unimelb.edu.au/student
ssh xiaojie@10.100.229.246
ssh xiaojie@10.100.228.181

conda create -n py27 python=2.7
conda create -n py34 python=3.4

# tensorflow tensorboard
export CUDA_VISIBLE_DEVICES=''
ssh -NL 6006:localhost:6006 xiaojie@10.100.228.181

# jingwei: extract image features by vgg16
cd jingwei/image_feature/matcovnet/
wget http://lixirong.net/data/csur2016/matconvnet-1.0-beta8.tar.gz
tar -xzvf matconvnet-1.0-beta8.tar.gz
wget http://lixirong.net/data/csur2016/matconvnet-models.tar
.gz
tar -xzvf matconvnet-models.tar.gz
matlab -nodisplay -nosplash -nodesktop -r "run('extract_vggnet.m');" # ds = 'yfcc2k';
matlab -nodisplay -nosplash -nodesktop -r "run('extract_vggnet.m');" # ds = 'yfcc8k';
zip -r survey_data.zip survey_data -x survey_data/yfcc8k/ImageData/\* survey_data/yfcc2k/ImageData/\*

# jingwei: precompute k nearest neighbors
cd jingwei/util/simpleknn/
sudo apt-get install libboost-dev
./build.sh

#
./do_knntagrel.sh yfcc8k yfcc2k vgg-verydeep-16-fc7relu

/home/xiaojie/Projects/data/yfcc100m/survey_data/yfcc2k/autotagging/yfcc2k/yfcc8k/concepts.txt/preknn/vgg-verydeep-16-fc7relu,cosineknn,5/id.tagvotes.txt









# do_tagprop.sh
./do_getknn.sh yfcc8k yfcc8k vgg-verydeep-16fc7relu 0 1 1
./do_getknn.sh yfcc8k yfcc2k vgg-verydeep-16fc7relu 0 1 1
# simple knn
conda install libgcc # ubuntu
brew install boost --c++11 # mac
./fasttext supervised -input yfcc10k/yfcc10k.train -output yfcc10k/model_yfcc10k -lr 1.0 -epoch 100
./fasttext test yfcc10k/model_yfcc10k.bin yfcc10k/yfcc10k.valid 5