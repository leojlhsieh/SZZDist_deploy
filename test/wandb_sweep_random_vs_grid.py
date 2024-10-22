# %%
import wandb
import time
import os
'''
template for sweep config on wandb website

method: random or grid
metric:
  goal: maximize
  name: test/accuracy
parameters:
  hp1:
    values: [3e-3, 6e-3, 12e-3]
  hp2:
    values: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  hp3:
    values: [150, 300, 600]
  hp4:
    values: [3, 4, 6]
'''

os.environ["WANDB_API_KEY"] = 'c08ff557ed402c26a0c57ca0e7803529bdba9268'
wandb.login()
# full_sweep_id = 'leohsieh-epfl/szzbpm-distil-5/honv7b1g'  # empty-random-honv7b1g
full_sweep_id = 'leohsieh-epfl/szzbpm-distil-5/kh1kceen'  # empty-grid-kh1kceen

def called_by_wandb_sweep():
    time_stamp = time.time_ns()
    with wandb.init(name=f'grid_sweep{time_stamp}'):
    # with wandb.init(name=f'random_sweep{time_stamp}'):
        print(wandb.config)
        wandb.log({'test/accuracy': time_stamp})
    return  # End of function


wandb.agent(full_sweep_id, called_by_wandb_sweep, count=3)
wandb.finish()
print("wandb.finish() completed!")



# Conclusion
'''
`andom` may have chance repeat the same combination, and keep running even if all combinations are done.
`grid` wlll cover all the combinations in order, and will stop sweeping when all combinations are done.
Leo decided to use `grid` to make sure no time is wasted on repeated combinations.
Leo implemented the random order grid sweep in the manually_random_grid.py.
'''