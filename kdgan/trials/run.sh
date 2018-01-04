python test.py \
    --model_name=vgg_16 \
    --checkpoint_exclude_scopes=vgg_16/fc8 \
    --trainable_scopes=vgg_16/fc8

# exit

# python test.py \
#     --model_name=resnet_v2_200 \
#     --checkpoint_exclude_scopes=resnet_v2_200/logits \
#     --trainable_scopes=resnet_v2_200/logits

# exit

python test.py \
    --model_name=nasnet_large \
    --checkpoint_exclude_scopes=final_layer/FC \
    --trainable_scopes=final_layer/FC

