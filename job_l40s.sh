#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 4:10:00
#SBATCH --gpus 1
#SBATCH --qos kuma
#SBATCH --partition l40s
# partition ['h100' or 'l40s']

# time 4:10:00 for 15 epochs of 128 batch size on L40S
# time 1:25:00 for 15 epochs of 128 batch size on H100
echo "==== Start ==================================="
nvcc --version
echo "======================================="
python ./tool/check_gpu.py
echo "======================================="
python ./leo_wandb_sweep_2.py --machine_name kuma_L40S --sweep_id hnlhb6il
echo "==== Done ==================================="

