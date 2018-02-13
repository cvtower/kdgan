proj_dir=$HOME/Projects
kdgan_dir=$proj_dir/kdgan_xw

checkpoint_dir=${kdgan_dir}/kdgan/checkpoints
pretrained_dir=${checkpoint_dir}/pretrained
datafig_dir=${kdgan_dir}/kdgan/datafigs
runs_dir=${kdgan_dir}/results/runs
pkls_dir=${kdgan_dir}/results/pkls

rootpath=${proj_dir}/data/yfcc100m/survey_data
codepath=${kdgan_dir}/jingwei

train_dataset=yfcc9k
test_dataset=yfcc0k
annotation_name=concepts.txt

filename=${train_dataset}_${test_dataset}_gen
gen_model_run=$runs_dir/$filename.run

python eval_model.py \
  --gen_model_ckpt=$checkpoint_dir/gen_vgg_16.ckpt \
  --gen_model_run=${gen_model_run} \
  --dataset=yfcc10k \
  --image_model=vgg_16 \
  --feature_size=4096