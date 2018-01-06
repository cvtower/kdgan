kdgan_dir=/home/xiaojie/Projects/kdgan/kdgan
pretrained_dir=$kdgan_dir/checkpoints/pretrained

python yfcc10k_fast.py \
    --model_name=inception_resnet_v2 \
    --preprocessing_name=inception_resnet_v2 \
    --checkpoint_path=$pretrained_dir/inception_resnet_v2_2016_08_30.ckpt \
    --end_point=global_pool

exit

python yfcc10k_fast.py \
    --model_name=vgg_16 \
    --preprocessing_name=vgg_16 \
    --checkpoint_path=$pretrained_dir/vgg_16.ckpt \
    --end_point=vgg_16/fc7
