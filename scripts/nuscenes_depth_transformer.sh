#!/bin/bash

srun python3 main.py --name nuscenes_depth_transformer --base configs/nuscenes_depth_transformer.yaml -t True --gpus 0, --max_epochs 20

