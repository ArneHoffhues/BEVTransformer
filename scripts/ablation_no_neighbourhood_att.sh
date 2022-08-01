#!/bin/bash

srun python3 main.py --name kitti360_ablation_no_neighbourhood_att --base configs/kitti360_ablation_no_neighbourhood_att.yaml -t True --gpus 0, --max_epochs 20

