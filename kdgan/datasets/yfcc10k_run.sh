kdgan_dir=/home/xiaojie/Projects/kdgan/kdgan
pretrained_dir=$kdgan_dir/checkpoints/pretrained

python yfcc10k_fast.py \
    --model_name=vgg_19 \
    --preprocessing_name=vgg_19 \
    --checkpoint_path=$pretrained_dir/vgg_19.ckpt \
    --end_point=vgg_19/fc7
# 4096
exit

python yfcc10k_fast.py \
    --model_name=resnet_v2_152 \
    --preprocessing_name=resnet_v2_152 \
    --checkpoint_path=$pretrained_dir/resnet_v2_152.ckpt \
    --end_point=global_pool
#2048
exit

python yfcc10k_fast.py \
    --model_name=inception_resnet_v2 \
    --preprocessing_name=inception_resnet_v2 \
    --checkpoint_path=$pretrained_dir/inception_resnet_v2_2016_08_30.ckpt \
    --end_point=global_pool
# 1536

python yfcc10k_fast.py \
    --model_name=vgg_16 \
    --preprocessing_name=vgg_16 \
    --checkpoint_path=$pretrained_dir/vgg_16.ckpt \
    --end_point=vgg_16/fc7
# 4096