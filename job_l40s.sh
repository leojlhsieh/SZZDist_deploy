#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 0:01:00
#SBATCH --gpus 1
#SBATCH --qos kuma
#SBATCH --partition l40s
# partition ['h100' or 'l40s']

# time 4:10:00 for 15 epochs of 128 batch size
echo "==== Start ==================================="
nvcc --version
echo "======================================="
python ./tool/check_gpu.py
echo "======================================="
python ./leo_wandb_sweep.py --machine_name kuma_L40S
echo "==== Done ==================================="

