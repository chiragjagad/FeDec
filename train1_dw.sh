#!/usr/bin/env bash
nvidia-smi
conda env create -f environment_test.yml
source activate abc
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
pip install tensorboard
python train_dw.py --arch 'arch_name_dw' 