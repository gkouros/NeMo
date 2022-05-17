#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate nemo

get_abs_filename() {
  # $1 : relative filename
  echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
}

ROOT=$(get_abs_filename "./")
DATAROOT="/esat/topaz/gkouros/datasets/"
EXPROOT="${ROOT}/exp/KITTI3D"

MESH_DIMENSIONS="single"
GPUS="0" #, 1, 2, 3, 4, 5, 6, 7"
# OCC_LEVEL="fully_visible"
# OCC_LEVEL="partly_occluded"
# OCC_LEVEL="largely_occluded"
OCC_LEVEL=""

PATH_KITTI3D="${DATAROOT}/KITTI3D"
PATH_PASCAL3DP="${DATAROOT}/pascal3d/PASCAL3D+_release1.1/"
TRAINED_NETWORK_PATH="${EXPROOT}/../PASCAL3D/NeMo_${MESH_DIMENSIONS}/"
SAVED_FEATURE_PATH="${EXPROOT}/Features_${MESH_DIMENSIONS}/"
SAVE_ACCURACY_PATH="${EXPROOT}/Accuracy_${MESH_DIMENSIONS}/"

PATH_CACHE_TESTING_SET="${DATAROOT}/KITTI3D/KITTI3D_val_NeMo/"

LOAD_FILE_NAME="saved_model_%s_799.pth"
SAVE_FEATURE_NAME="saved_feature_%s_%s.npz"

# Feature extraction
BATCH_SIZE=1

# Pose optimization
LEARNING_RATE=0.05
TOTAL_EPOCHS=300

IFS=', ' read -r -a GPU_LIST <<< "${GPUS}"
# ALL_CATEGORIES=("aeroplane"  "bicycle"  "boat"  "bottle"  "bus"  "car"  "chair"  "diningtable"  "motorbike"  "sofa"  "train"  "tvmonitor")
ALL_CATEGORIES=("car")

# if [ "${#GPU_LIST[@]}" -eq "8" ]; then
#     # 8 GPU SETTING
#     # GPU_ASSIGNMENT=("0"       "1"        "2"     "1"       "3"    "4"    "3"      "5"            "6"          "6"     "7"      "7")
#     GPU_ASSIGNMENT=("${GPU_LIST[0]}" "${GPU_LIST[1]}" "${GPU_LIST[2]}" "${GPU_LIST[1]}" "${GPU_LIST[3]}" "${GPU_LIST[4]}" "${GPU_LIST[3]}" "${GPU_LIST[5]}" "${GPU_LIST[6]}" "${GPU_LIST[6]}" "${GPU_LIST[7]}" "${GPU_LIST[7]}")
# else
#     # 4 GPU SETTING
#     # GPU_ASSIGNMENT=("0"       "0"        "1"     "0"       "1"    "3"    "1"      "2"            "2"          "2"     "1"      "2")
#     GPU_ASSIGNMENT=("${GPU_LIST[0]}" "${GPU_LIST[0]}" "${GPU_LIST[1]}" "${GPU_LIST[0]}" "${GPU_LIST[1]}" "${GPU_LIST[3]}" "${GPU_LIST[1]}" "${GPU_LIST[2]}" "${GPU_LIST[2]}" "${GPU_LIST[2]}" "${GPU_LIST[1]}" "${GPU_LIST[2]}")
# fi
GPU_ASSIGNMENT=("${GPU_LIST[0]}" "${GPU_LIST[0]}")


for CATEGORY in "${ALL_CATEGORIES[@]}"; do
    mesh_path="${PATH_PASCAL3DP}/CAD_%s/%s/"
    CUDA_VISIBLE_DEVICES="${GPUS}" python "${ROOT}/code/ExtractFeatures.py" \
            --mesh_path "${mesh_path}" \
            --mesh_d "${MESH_DIMENSIONS}" \
            --save_dir "${TRAINED_NETWORK_PATH}" \
            --type_ "${CATEGORY}" \
            --ckpt "${LOAD_FILE_NAME}" \
            --data_pendix "${OCC_LEVEL}"\
            --root_path "${PATH_CACHE_TESTING_SET}" \
            --save_features_path "${SAVED_FEATURE_PATH}" \
            --save_features_name "${SAVE_FEATURE_NAME}" \
            --batch_size $BATCH_SIZE
done

mkdir "${SAVE_ACCURACY_PATH}" >/dev/null 2>&1

for ((i=0;i<${#ALL_CATEGORIES[@]};++i)); do
    CUDA_VISIBLE_DEVICES="${GPU_ASSIGNMENT[$i]}" python "${ROOT}/code/MeshPoseSolveAll.py" \
            --type_ "${ALL_CATEGORIES[$i]}" --mesh_d "${MESH_DIMENSIONS}" \
            --mesh_path "${PATH_PASCAL3DP}/CAD_%s/%s/" \
            --mesh_path_ref "${PATH_PASCAL3DP}/CAD/%s/" \
            --feature_path "${SAVED_FEATURE_PATH}" \
            --feature_name "${SAVE_FEATURE_NAME}" \
            --data_pendix "${OCC_LEVEL}" \
            --save_accuracy "${SAVE_ACCURACY_PATH}/${ALL_CATEGORIES[$i]}${OCC_LEVEL}" \
            --anno_path "${PATH_CACHE_TESTING_SET}/annotations/%s/" \
            --total_epochs $TOTAL_EPOCHS \
            --lr $LEARNING_RATE
done

wait
python "${ROOT}/code/CalAccuracy.py" \
    --load_accuracy "${SAVE_ACCURACY_PATH}" \
    --data_pendix "${OCC_LEVEL}"

conda deactivate