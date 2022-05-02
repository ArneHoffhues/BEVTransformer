#!/bin/bash

srun python3 main.py --name pon_nuscenes_default_res --base configs/pon_nuscenes_with_decoder.yaml -t True --gpus 0,1 --max_epochs 20

