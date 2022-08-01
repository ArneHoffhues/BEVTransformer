#!/bin/bash

srun python3 main.py --name kitti360_depth_encoding_6layers --base configs/kitti360_with_decoder_and_depth_encoding.yaml -t True --gpus 0, --max_epochs 20

