#!/bin/bash

module load gcc/13.2.0 python/3.11.7 tmux/3.4 cuda/12.4.1
source /scratch/jlhsieh/leo_scratch/venv-leo/bin/activate
module list


cd /scratch/jlhsieh/leo_scratch/SZZDist_deploy
sbatch /scratch/jlhsieh/leo_scratch/SZZDist_deploy/job_01_l40s.sh
sbatch --qos=kuma --nodes=1  --ntasks=1 --cpus-per-task=1 --gpus=1 --mem=4G --time=0:30:00 --partition=h100 /scratch/jlhsieh/leo_scratch/SZZDist_deploy/job_02_h100.sh
Squeue



git clone https://github.com/leojlhsieh/SZZDist_deploy.git
git pull

Sinteract --qos=kuma --nodes=1  --ntasks=1 --cpus-per-task=1 --gpus=1 --mem=4G --time=0:30:00 --partition=h100
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


python play2.py --a=apple --b=99






# choices=[1, 2, 3, 4, 5, 6, 7, 8]
# choices=[75, 150, 300, 600, 1200])
# choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'])
# choices=['musta_3090Ti', 'musta_2080Ti', 'haitao_2080Ti', 'kuma_L40S', 'kuma_H100'])



python.exe leo_wandb_sweep_4.py \
    --sweep_id=8li9kx8l \
    --machine_name=kuma_H100 \
    --epochs=2 \
    --loss_ratio=0.9 \
    --data_name=my_cifar10 \
    --small_toy=10 \
    --batch_size=4 \
    --bpm_depth=8 \
    --bpm_width=1200 \

# choices=[1, 2, 3, 4, 5, 6, 7, 8]
# choices=[75, 150, 300, 600, 1200])
# choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'])
# choices=['musta_3090Ti', 'musta_2080Ti', 'haitao_2080Ti', 'kuma_L40S', 'kuma_H100'])

