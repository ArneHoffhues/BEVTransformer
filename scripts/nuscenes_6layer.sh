#!/bin/bash

srun python3 main.py --name nuscenes_priors_6layers --base configs/nuscenes_with_decoder_priors.yaml -t True --gpus 0,1 --max_epochs 20

