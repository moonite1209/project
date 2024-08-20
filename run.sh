#!/bin/bash
dataset_name=$1
feature_level=$2
language_fature_3d=$3

if [ "$#" -eq 0 ]; then
    echo "Usage: bash run.sh {dataset_name: string} {feature_level: int} {language_fature_3d: \"3d\" | \"\"}"
# image
elif [ "$#" -eq 1 ]; then
    # convert
    python train.py -s data/${dataset_name}/ -m output/${dataset_name}/ --eval # train gaussians
    python render.py -m output/${dataset_name}/
# language feature
elif [ "$#" -eq 2 ]; then
    # convert
    python train.py -s data/${dataset_name}/ -m data/${dataset_name}/output/${dataset_name}/ --eval # train gaussians
    python preprocess.py --dataset_path data/${dataset_name}  # get segment map and segment semantics
    python autoencoder/train.py --dataset_name ${dataset_name} --dataset_path data/${dataset_name} # preprocess encode 512-dim to 3-dim
    python autoencoder/test.py --dataset_name ${dataset_name} --dataset_path data/${dataset_name} # preprocess encode 512-dim to 3-dim
    python train.py -s data/${dataset_name} -m output/${dataset_name}_${feature_level} --start_checkpoint data/${dataset_name}/output/${dataset_name}/chkpnt30000.pth --feature_level ${feature_level} --mode langsplat --eval # train semantics
    python render.py -m output/${dataset_name}_${feature_level}/ --feature_level ${feature_level} --mode langsplat
# language feature 3d
elif [ "$#" -eq 3 ]; then
    # convert
    python train.py -s data/${dataset_name}/ -m data/${dataset_name}/output/${dataset_name}/ --eval # train gaussians
    python preprocess.py --dataset_path data/${dataset_name}  # get segment map and segment semantics
    python autoencoder/train.py --dataset_name ${dataset_name} --dataset_path data/${dataset_name} # preprocess encode 512-dim to 3-dim
    python autoencoder/test.py --dataset_name ${dataset_name} --dataset_path data/${dataset_name} # preprocess encode 512-dim to 3-dim
    python train.py -s data/${dataset_name} -m output/${dataset_name}_3d_${feature_level} --feature_level ${feature_level} --mode ours --eval # train semantics
    python render.py -m output/${dataset_name}_3d_${feature_level}/ --feature_level ${feature_level} --mode ours
    python eval/evaluate_iou_loc.py --dataset_name ${dataset_name} --feat_dir output/${dataset_name} --ae_ckpt_dir ckpt --output_dir eval_result --json_folder data/lerf_ovs/label/ --mode ours
fi