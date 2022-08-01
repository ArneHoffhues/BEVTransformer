#!/bin/bash

srun python3 main.py --name nuscenes_original_pbev --base configs/nuscenes_original_pbev.yaml -t True --gpus 0, --max_epochs 30

