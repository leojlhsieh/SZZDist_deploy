# Description: Run a python script with arguments                                                             .                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

& C:/Users/musta/miniconda3/envs/leopy311/python.exe `
 c:/Users/musta/Documents/leo-code-space/SZZDist_deploy/leo_wandb_sweep_5.py `
    --machine_name=musta_3090Ti `
    --data_name=my_mnist `
    --epochs=2 `
    --small_toy=0 `
    --batch_size=32 



# Note:
    # hyperparameters = {
    #     'bpm_depth': [3, 4, 6],  # Number of SLM modulation (aka bounce) in BPM
    #     'bpm_width': [150, 300, 600],  # Width of SLM [pixel], height is same as width
    #     'Ldist': [3e-3, 6e-3, 12e-3],  # Distance between SLM & mirror [m]
    #     'lr_bpm': log_uniform_distribution(start=5e-3 * 0.1, end=5e-3 * 10, points=7),  # nPOLO paper use lr=5e-3 for optics
    #     'lr_class': log_uniform_distribution(start=2.5e-4 * 0.1, end=2.5e-4 * 10, points=7),  # nPOLO paper use lr=2.5e-4 for digital classifier
    #     'loss_ratio': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Total loss = loss_ratio * L1 + (1-loss_ratio) * L2
    # }


# Test time spend
# & C:/Users/musta/miniconda3/envs/leopy311/python.exe `
#  c:/Users/musta/Documents/leo-code-space/SZZDist_deploy/leo_wandb_sweep_5.py `
#     --machine_name=musta_3090Ti `
#     --data_name=my_imagenette `
#     --epochs=2 `
#     --small_toy=0 `
#     --batch_size=32 `
#     --bpm_depth=3 `
#     --bpm_width=150 `
#     --sweep_id=honv7b1g


# choices=[1, 2, 3, 4, 5, 6, 7, 8]
# choices=[75, 150, 300, 600, 1200])
# choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette'])
# choices=['musta_3090Ti', 'musta_2080Ti', 'haitao_2080Ti', 'kuma_L40S', 'kuma_H100'])



