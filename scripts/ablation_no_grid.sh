#!/bin/bash

srun python3 main.py --name kitti360_ablation_no_grid --base configs/kitti360_ablation_no_grid.yaml -t True --gpus 0, --max_epochs 20

