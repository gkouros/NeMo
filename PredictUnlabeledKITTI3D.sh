#!/bin/bash

python3 code/PredUnlabeledDataset.py \
    --img_path="/esat/topaz/gkouros/datasets/KITTI3D_NeMo/KITTI_train_distcrop/images/car" \
    --ckpt="exp/NeMo_single/saved_model_car_799.pth" \
    --vertical_shift=0 \
    --tar_horizontal_size=256 \
    --mesh_path="/esat/topaz/gkouros/datasets/pascal3d/PASCAL3D+_release1.1/CAD_multi/car/"