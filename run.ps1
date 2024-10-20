# Description: Run a python script with arguments                                                             .                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
# & C:/Users/musta/miniconda3/envs/leopy311/python.exe `
#  c:/Users/musta/Documents/leo-code-space/SZZDist_deploy/leo_wandb_sweep_3_no_wandb.py `
#     --epochs 1 `
#     --small_toy 7 `
#     --num_cpu_worker 0 
 

& C:/Users/musta/miniconda3/envs/leopy311/python.exe `
 c:/Users/musta/Documents/leo-code-space/SZZDist_deploy/leo_wandb_sweep_4.py `
    --sweep_id=8li9kx8l `
    --machine_name=musta_3090Ti `
    --small_toy=10 `
    --loss_ratio=0.8 `
    --epochs=4 `
    --data_name=my_cifar10 `
    --batch_size=4 `
    --bpm_depth=8 `
    --bpm_width=1200 `

# choices=[1, 2, 3, 4, 5, 6, 7, 8]
# choices=[75, 150, 300, 600, 1200])
# choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'])
# choices=['musta_3090Ti', 'musta_2080Ti', 'haitao_2080Ti', 'kuma_L40S', 'kuma_H100'])



# & C:/Users/musta/miniconda3/envs/leopy311/python.exe `
#  C:\Users\musta\Documents\leo-code-space\SZZDist_deploy\play.py


