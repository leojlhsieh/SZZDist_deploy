#!/bin/bash
#SBATCH --qos kuma
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus 1
#SBATCH --mem 8G
#SBATCH --time 00:20:00
#SBATCH --partition h100
# partition ['h100' or 'l40s']

# time 4:10:00 for 15 epochs of 128 batch size on L40S
# time 1:25:00 for 15 epochs of 128 batch size on H100
echo "==== Start ==================================="
python /scratch/jlhsieh/leo_scratch/SZZDist_deploy/leo_wandb_sweep_4.py \
    --sweep_id=8li9kx8l \
    --machine_name=kuma_H100 \
    --epochs=5 \
    --lr_bpm=5e-4 \
    --lr_feature=1e-4 \
    --lr_class=1e-4 \
    --loss_ratio=0.9 \
    --data_name=my_imagenette \
    --small_toy=0 \
    --batch_size=32 \
    --bpm_depth=1 \
    --bpm_width=75 \

# empty4 - 8li9kx8l
# choices=[1, 2, 3, 4, 5, 6, 7, 8]
# choices=[75, 150, 300, 600, 1200])
# choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'])
# choices=['musta_3090Ti', 'musta_2080Ti', 'haitao_2080Ti', 'kuma_L40S', 'kuma_H100'])
echo "==== Done ==================================="

