#!/bin/bash

srun python3 main.py --name kitti360_depth_addition_6layers --base configs/kitti360_with_decoder_and_depth.yaml -t True --gpus 0, --max_epochs 20

