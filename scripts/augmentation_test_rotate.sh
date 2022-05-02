#!/bin/bash

srun python3 main.py --name kitti360_augmentation_rotate --base configs/kitti360_with_decoder_small_rotate.yaml -t True --gpus 0,1 --max_epochs 60

