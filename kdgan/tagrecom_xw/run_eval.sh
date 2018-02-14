proj_dir=$HOME/Projects
kdgan_dir=$proj_dir/kdgan_xw
runs_dir=${kdgan_dir}/results/runs
pkls_dir=${kdgan_dir}/results/pkls

train_dataset=yfcc9k
test_dataset=yfcc0k

python eval_model.py \
  --gen_checkpoint_dir=$proj_dir/kdgan_ys/kdgan/checkpoints/kdgan_allit \
  --gen_model_run=${runs_dir}/${train_dataset}_${test_dataset}_kdgan.run \
  --dataset=yfcc10k \
  --image_model=vgg_16 \
  --feature_size=4096
exit


python eval_model.py \
  --gen_model_ckpt=${kdgan_dir}/kdgan/checkpoints/gen_vgg_16.ckpt \
  --gen_model_run=${runs_dir}/${train_dataset}_${test_dataset}_gen.run \
  --dataset=yfcc10k \
  --image_model=vgg_16 \
  --feature_size=4096
exit