# %%
import wandb
import time
import os




os.environ["WANDB_API_KEY"] = 'c08ff557ed402c26a0c57ca0e7803529bdba9268'
wandb.login()
full_sweep_id = 'leohsieh-epfl/szzbpm-distil-5/honv7b1g'  # empty-random-honv7b1g


def called_by_wandb_sweep_agent():
    time_stamp = time.time_ns()
    with wandb.init(config={'my message': 'Hello, World!'}, name=f'grid_sweep{time_stamp}'):
    # with wandb.init(name=f'random_sweep{time_stamp}'):
        print(wandb.config)
        wandb.log({'test/accuracy': time_stamp})
        wandb.config.update({'my message': 'I am Leo!'})
        wandb.config.update({'hp1': 100000000000})
        print(wandb.config)
    return  # End of function


wandb.agent(full_sweep_id, called_by_wandb_sweep_agent, count=1)
wandb.finish()
print("wandb.finish() completed!")
