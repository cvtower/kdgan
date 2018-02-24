proj_dir=$HOME/Projects
kdgan_dir=$proj_dir/kdgan_xw
runs_dir=${kdgan_dir}/results/runs
pkls_dir=${kdgan_dir}/results/pkls

dataset=yfcc10k
image_model=vgg_16
feature_size=4096
train_dataset=yfcc9k
test_dataset=yfcc0k

python eval_model.py \
  --dataset=$dataset \
  --image_model=${image_model} \
  --feature_size=${feature_size} \
  --model_name=gen \
  --model_ckpt=${kdgan_dir}/kdgan/checkpoints/gen_vgg_16.ckpt \
  --model_run=${runs_dir}/${train_dataset}_${test_dataset}_gen.run
exit


