#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --time 0:01:00
#SBATCH --gpus=1
#SBATCH --partition=l40s
#SBATCH --qos=kuma

# partition is h100 or l40s
nvcc --version
echo "======================================="
echo "leo leo"
echo "======================================="
python /scratch/jlhsieh/leo_scratch/leo-code-space/check_gpu.py
echo "======================================="
python /scratch/jlhsieh/leo_scratch/leo-code-space/play.py


