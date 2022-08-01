#!/bin/bash

srun python3 main.py --name kitti360_augmentation_no_augmentation_6layers --base configs/kitti360_with_decoder_small.yaml -t True --gpus 0, --max_epochs 60

