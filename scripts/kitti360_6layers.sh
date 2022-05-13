#!/bin/bash

#SBATCH --partition=trtx-lo
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=160:00:00
#SBATCH --pty
#SBATCH --cpus-per-task=4

export PL_FAULT_TOLERANT_TRAINING=1

srun python3 main.py --name kitti360_priors_6layers --base configs/kitti360_with_decoder_priors.yaml -t True --gpus 0, --max_epochs 20

