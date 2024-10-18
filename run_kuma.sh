#!/bin/bash

module load gcc/13.2.0 python/3.11.7 tmux/3.4 cuda/12.4.1
source /scratch/jlhsieh/leo_scratch/venv-leo/bin/activate
module list


cd /scratch/jlhsieh/leo_scratch/SZZDist_deploy
sbatch /scratch/jlhsieh/leo_scratch/SZZDist_deploy/job_kuma_h100.sh
sbatch /scratch/jlhsieh/leo_scratch/SZZDist_deploy/job_kuma_l40s.sh
Squeue