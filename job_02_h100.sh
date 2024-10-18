#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 0:09:00
#SBATCH --gpus 1
#SBATCH --qos kuma
#SBATCH --partition h100
# partition ['h100' or 'l40s']

nvcc --version
echo "======================================="
echo "leo leo"
echo "======================================="
python /scratch/jlhsieh/leo_scratch/SZZDist_deploy/tool/check_gpu.py
echo "======================================="
python /scratch/jlhsieh/leo_scratch/SZZDist_deploy/leo_wandb_sweep.py --machine_name kuma_H100

