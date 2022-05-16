#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nemo

get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT=$(get_abs_filename "./")
DATAROOT="${ROOT}/kdata"
EXPROOT="${ROOT}/exp/KITTI3D"

MESH_DIMENSIONS="single"
GPUS="0" #, 1, 2, 3, 4, 5, 6"

PATH_CACHE_TRAINING_SET="${DATAROOT}/KITTI3D_train_NeMo/"
SAVED_NETWORK_PATH="${EXPROOT}/NeMo_${MESH_DIMENSIONS}/"
MESH_PATH="${ROOT}/data/PASCAL3D+_release1.1/CAD_%s/%s/"

BATCH_SIZE=108
TOTAL_EPOCHS=800
LEARNING_RATE=0.0001
WEIGHT_CLUTTER=5e-3
NUM_CLUTTER_IMAGE=5
NUM_CLUTTER_GROUP=512

CUDA_VISIBLE_DEVICES="${GPUS}" python "${ROOT}/code/TrainNeMoKITTI3D.py" \
        --mesh_path "${MESH_PATH}" --save_dir "${SAVED_NETWORK_PATH}" \
        --type_ "car" --root_path "${PATH_CACHE_TRAINING_SET}" --mesh_d "${MESH_DIMENSIONS}" \
        --sperate_bank "False" --batch_size $BATCH_SIZE --total_epochs $TOTAL_EPOCHS \
        --lr $LEARNING_RATE --weight_noise $WEIGHT_CLUTTER --num_noise $NUM_CLUTTER_IMAGE \
        --max_group $NUM_CLUTTER_GROUP

conda deactivate
