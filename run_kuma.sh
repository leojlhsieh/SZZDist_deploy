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
