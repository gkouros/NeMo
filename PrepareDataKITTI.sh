#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nemo

dir=/users/visics/gkouros/projects/NeMo
cd $dir

get_abs_filename() {
    # $1 : relative filename
    echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT=$(get_abs_filename "./")

ENABLE_OCCLUDED=true
DATAROOT="${ROOT}/kdata"

PATH_PASCAL3DP="${ROOT}/data/PASCAL3D+_release1.1/"
PATH_KITTI3D="${DATAROOT}"
PATH_CACHE_TRAINING_SET="${PATH_KITTI3D}/KITTI3D_train_NeMo/"
PATH_CACHE_TESTING_SET="${PATH_KITTI3D}/KITTI3D_val_NeMo/"

echo $PATH_KITTI3D

####################################################################################################
# assumes datasets already downloaded

####################################################################################################
# Run dataset creator
echo "Create raw KITTI3D dataset!"

python3 code/dataset/CreateKITTI3D.py --split train --datadir $DATAROOT
python3 code/dataset/CreateKITTI3D.py --split val --datadir $DATAROOT

####################################################################################################
# # Create 3D annotations

# generate 3D annotations for training set
python3 ./code/dataset/generate_3Dkitti3D.py \
    --overwrite False \
    --root_path "${PATH_CACHE_TRAINING_SET}" \
    --mesh_path "${PATH_PASCAL3DP}"

# generate 3D annotations for test set
python3 ./code/dataset/generate_3Dkitti3D.py \
    --overwrite False \
    --root_path "${PATH_CACHE_TESTING_SET}" \
    --mesh_path "${PATH_PASCAL3DP}"

wait

# deactivate virtual environment
conda deactivate
