#!/bin/bash

srun python3 main.py --name nuscenes_depth_unprojection --base configs/nuscenes_depth_combined_loss_full_res.yaml -t True --gpus 0,1 --max_epochs 20

