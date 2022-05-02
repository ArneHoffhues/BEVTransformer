#!/bin/bash

srun python3 main.py --name kitti360_augmentation_bottom_crop --base configs/kitti360_with_decoder_small_bottom_crop.yaml -t True --gpus 0,1 --max_epochs 60

