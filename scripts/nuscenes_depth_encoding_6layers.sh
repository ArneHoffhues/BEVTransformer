#!/bin/bash

srun python3 main.py --name nuscenes_depth_encoding_6layers --base configs/nuscenes_with_decoder_and_depth_encoding.yaml -t True --gpus 0,1 --max_epochs 20

