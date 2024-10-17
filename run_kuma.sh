#!/bin/bash

module load gcc/13.2.0 python/3.11.7 tmux/3.4 cuda/12.4.1
source /scratch/jlhsieh/leo_scratch/venv-leo/bin/activate
module list
cd /scratch/jlhsieh/leo_scratch/leo-code-space
sbatch /scratch/jlhsieh/leo_scratch/leo-code-space/job_leo.sh 
Squeue
