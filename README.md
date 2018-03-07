# kdgan
pip install -Ue .

remote.unimelb.edu.au/student
ssh xiaojie@10.100.229.246 # cpu
ssh xiaojie@10.100.228.151 # gpu cy
ssh xiaojie@10.100.228.149 # gpu cz
ssh xiaojie@10.100.228.181 # gpu xw
ssh xiaojie@10.100.228.28 # gpu yz
ssh xiaojiewang@10.100.228.28 # gpu yz # initialpassword

# bank
5217291828507288

# text classification
git reset --hard 7c0564610815732283cc968c387d4b000fa38a68

conda create -n py27 python=2.7
conda create -n py34 python=3.4

# tensorflow tensorboard
export CUDA_VISIBLE_DEVICES=''
ssh -NL 6006:localhost:6006 xiaojie@10.100.229.246 # cpu
ssh -NL 6006:localhost:6006 xiaojie@10.100.228.181 # gpu

python mnist_bn_wi.py --weight-init xavier --bias-init zero --batch-norm True

virtualenv --system-site-packages venv
pip install --ignore-installed --upgrade tensorflow
pip install --ignore-installed -r requirements.txt

################################################################
#
# baseline
#
################################################################

# jingwei: extract image features by vgg16
cd jingwei/image_feature/matcovnet/
wget http://lixirong.net/data/csur2016/matconvnet-1.0-beta8.tar.gz
tar -xzvf matconvnet-1.0-beta8.tar.gz
wget http://lixirong.net/data/csur2016/matconvnet-models.tar.gz
tar -xzvf matconvnet-models.tar.gz
matlab -nodisplay -nosplash -nodesktop -r "run('extract_vggnet.m');"
# jingwei: precompute k nearest neighbors
conda install libgcc # ubuntu
brew install boost --c++11 # mac
cd jingwei/util/simpleknn/
sudo apt-get install libboost-dev
./build.sh
# jingwei: knn
./do_knntagrel.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu
# jingwei: tagprop
import nltk & nltk.download('wordnet')
./do_getknn.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu 0 1 1
./do_getknn.sh yfcc9k yfcc9k vgg-verydeep-16-fc7relu 0 1 1
setup-tagprop.sh
wget http://lear.inrialpes.fr/people/guillaumin/code/TagProp_0.2.tar.gz
./do_tagprop.sh yfcc9k yfcc0k vgg-verydeep-16-fc7relu
patch TagProp/sigmoids.m < sigmoids.m.patch
patch TagProp/tagprop_learn.m < tagprop_learn.m.patch
patch TagProp/tagprop_predict.m < tagprop_predict.m.patch
cd TagProp & matlab -nodesktop -nosplash -r "mex tagpropCmt.c; exit"
# jingwei: evaluation
./eval_pickle.sh yfcc0k

################################################################
#
# model compression
#
################################################################

python download_and_convert_data.py \
  --dataset_name=mnist \
  --dataset_dir=$HOME/Projects/data/mnist

python train_image_classifier.py \
  --train_dir=$HOME/Projects/kdgan/kdgan/slimmodels \
  --dataset_name=mnist \
  --dataset_split_name=train \
  --dataset_dir=$HOME/Projects/data/mnist \
  --model_name=lenet

python eval_image_classifier.py \
  --alsologtostderr \
  --checkpoint_path=$HOME/Projects/kdgan/kdgan/slimmodels \
  --dataset_dir=$HOME/Projects/data/mnist \
  --dataset_name=mnist \
  --dataset_split_name=test \
  --model_name=lenet

# cifar 10
http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130
https://github.com/tensorflow/models/tree/master/official/resnet
https://github.com/BIGBALLON/cifar-10-cnn
https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
https://github.com/shmsw25/cifar10-classification
https://github.com/ethereon/caffe-tensorflow

# mnist
https://github.com/clintonreece/keras-cloud-ml-engine
https://github.com/keras-team/keras/tree/master/examples

https://github.com/hwalsuklee/how-far-can-we-go-with-MNIST
http://www.pythonexample.com/user/vamsiramakrishnan

# gan trick
https://github.com/gitlimlab/SSGAN-Tensorflow


https://github.com/xiaojiew1/kdgan/tree/bb88661286b092f3576bf2a7b58344ea503452f8/kdgan/mdlcompr/trials