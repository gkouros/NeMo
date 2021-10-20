#!/bin/bash
git clone https://github.com/Angtian/NeMo.git
cd NeMo
conda deactivate
conda create -n nemo python=3.7
conda activate nemo
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
