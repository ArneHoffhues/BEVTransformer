#!/bin/bash

srun python3 main.py --name kitti360_priors_6layers --base configs/kitti360_with_decoder_priors.yaml -t True --gpus 0,1 --max_epochs 20

