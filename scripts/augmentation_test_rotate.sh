#!/bin/bash

srun python3 main.py --name kitti360_augmentation_rotate_6layers --base configs/kitti360_with_decoder_small_rotate.yaml -t True --gpus 0, --max_epochs 60

