#!/bin/bash
module load gcc/13.2.0 python/3.11.7 tmux/3.4 cuda/12.4.1
source /scratch/jlhsieh/leo_scratch/venv-leo/bin/activate
module list

cd /scratch/jlhsieh/leo_scratch/SZZDist_deploy
for i in {1..24}; do
sbatch --time=00:01:00 --mem=8G --partition=h100  /scratch/jlhsieh/leo_scratch/SZZDist_deploy/job_leo.sh
sleep 2
Squeue
done


git clone https://github.com/leojlhsieh/SZZDist_deploy.git
git pull

Sinteract -c 1 -n 1 -t 00:15:00 -m 8G -p h100 -q kuma
Sinteract -c 1 -n 1 -t 00:30:00 -m 6G -p h100 -q kuma
usage: Sinteract [-c cores] [-n tasks] [-t time] [-m memory] [-p partition] [-a account] [-q qos] [-g resource] [-r reservation] [-s constraints]
options:
  -c cores       cores per task (default: 1)
  -n tasks       number of tasks (default: 1)
  -t time        as hh:mm:ss (default: 00:30:00)
  -m memory      as #[K|M|G] (default: 4G)
  -p partition   (default: )
  -a account     (default: lapd)
  -q qos         as [normal|gpu|gpu_free|mic|...] (default: )
  -g resource    as [gpu|mic][:count] (default is empty)
  -r reservation reservation name (default is empty)
  -s contraints  list of required features (default is empty)
                     Deneb/Eltanin: E5v2, E5v3
                     Fidis/Gacrux: E5v4, s6g1, mem128gb, mem192gb, mem256gb







#!/bin/bash

module load gcc/13.2.0 python/3.11.7 tmux/3.4 cuda/12.4.1
source /scratch/jlhsieh/leo_scratch/venv-leo/bin/activate
module list

mkdir -p /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/
cd /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/


# H100
for i in {1..100}; do
    # sbatch --time=01:56:00 --mem=4G --partition=h100  /scratch/jlhsieh/leo_scratch/jobH_mnist.sh
    # sbatch --time=01:40:00 --mem=4G --partition=h100  /scratch/jlhsieh/leo_scratch/jobH_cifar.sh
    # sbatch --time=01:55:00 --mem=4G --partition=h100  /scratch/jlhsieh/leo_scratch/jobH_fashion.sh
    sbatch --time=00:25:00 --mem=6G --partition=h100  /scratch/jlhsieh/leo_scratch/jobH_imgnet.sh
done

for i in {1..20}; do
    # sbatch --time=01:56:00 --mem=4G --partition=h100  /scratch/jlhsieh/leo_scratch/jobH_mnist.sh
    sbatch --time=01:40:00 --mem=4G --partition=h100  /scratch/jlhsieh/leo_scratch/jobH_cifar.sh
    # sbatch --time=01:55:00 --mem=4G --partition=h100  /scratch/jlhsieh/leo_scratch/jobH_fashion.sh
    # sbatch --time=00:25:00 --mem=6G --partition=h100  /scratch/jlhsieh/leo_scratch/jobH_imgnet.sh
done




# L40S
# for i in {1..300}; do
#     sbatch --time=01:00:00 --mem=6G --partition=l40s  /scratch/jlhsieh/leo_scratch/SZZDist_deploy/jobL_imgnet.sh
# done


# for i in {1..200}; do
#     sbatch --time=05:41:00 --mem=4G --partition=l40s  /scratch/jlhsieh/leo_scratch/SZZDist_deploy/jobL_mnist.sh
#     sbatch --time=04:45:00 --mem=4G --partition=l40s  /scratch/jlhsieh/leo_scratch/SZZDist_deploy/jobL_cifar.sh
#     sbatch --time=05:40:00 --mem=4G --partition=l40s  /scratch/jlhsieh/leo_scratch/SZZDist_deploy/jobL_fashion.sh
# done


# for i in {1..300}; do
#     sbatch --time=01:00:00 --mem=6G --partition=l40s  /scratch/jlhsieh/leo_scratch/SZZDist_deploy/jobL_imgnet.sh
# done

# grep -rl "sweep_index = 47" /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log | xargs grep -l "my_fashion_mnist"
# grep -rl "sweep_index = 47" /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log | xargs grep -l "my_fashion_mnist"
# grep -rl "sweep_index = 114" /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log | xargs grep -l "my_imagenette"

grep -rl "Test loss"

# find /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/wandb -type d -name "*ql1ed5zv*"
# wandb sync /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/wandb/run-20241023_050014-ql1ed5zv


# find /scratch/jlhsieh/leo_scratch -type d -name "*uz9pvfo9*"
# wandb sync --id kk22jwg8 /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/wandb/run-20241022_234653-kk22jwg8/files/output.log
# wandb sync --id kk22jwg8 /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/wandb/run-20241022_234653-kk22jwg8/files/output.log
# wandb sync --id kk22jwg8 /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/wandb/run-20241022_234653-kk22jwg8/files/output.log
# wandb sync --no-mark-synced /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/wandb/run-20241022_234653-kk22jwg8/

# wandb sync --no-mark-synced /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/wandb/run-20241023_151311-ei6g966q/
# wandb sync /scratch/jlhsieh/leo_scratch/SZZDist_deploy/slurm-log/wandb/run-20241023_152945-vogs17ze/

# mkdir -p /scratch/jlhsieh/leo_scratch/temp # p means parent directory, so it will create the parent directory if it does not exist
# echo "Directory '/scratch/jlhsieh/leo_scratch/temp' created or already exists."


# for i in {1..50}; do
#     git clone https://github.com/leojlhsieh/SZZDist_deploy.git "/scratch/jlhsieh/leo_scratch/temp/SZZDist_deploy_$(printf "%03d" $i)"
#     cd "/scratch/jlhsieh/leo_scratch/temp/SZZDist_deploy_$(printf "%03d" $i)"
#     cp /scratch/jlhsieh/leo_scratch/all_data/my_cifar10.tar.gz ./data/
#     sbatch ./job_l40s.sh
#     # sbatch ./job_h100.sh
#     sleep 2
#     Squeue
# done


# sbatch --time=2-00:00:00 moojob.run





# for i in {1..50}; do
#     git clone https://github.com/leojlhsieh/SZZDist_deploy.git "/scratch/jlhsieh/leo_scratch/temp/SZZDist_deploy_$(printf "%03d" $i)"
#     cd "/scratch/jlhsieh/leo_scratch/temp/SZZDist_deploy_$(printf "%03d" $i)"
#     cp /scratch/jlhsieh/leo_scratch/all_data/my_cifar10.tar.gz ./data/
#     sbatch ./job_l40s.sh
#     # sbatch ./job_h100.sh
#     sleep 2
#     Squeue
# done


# for i in {51..70}; do
#     git clone https://github.com/leojlhsieh/SZZDist_deploy.git "/scratch/jlhsieh/leo_scratch/temp/SZZDist_deploy_$(printf "%03d" $i)"
#     cd "/scratch/jlhsieh/leo_scratch/temp/SZZDist_deploy_$(printf "%03d" $i)"
#     cp /scratch/jlhsieh/leo_scratch/all_data/my_cifar10.tar.gz ./data/
#     # sbatch ./job_l40s.sh
#     sbatch ./job_h100.sh
#     sleep 2
#     Squeue
# done







