#!/bin/bash

srun python3 main.py --name kitti360_original_pbev_a40 --base configs/kitti360_original_pbev.yaml -t True --gpus 0, --max_epochs 20

