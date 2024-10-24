# %%
import os
from tqdm.auto import tqdm
import time
from argparse import ArgumentParser
import wandb
parser = ArgumentParser()
parser.add_argument("-v", "--verbose", action="store_true", default=True)
parser.add_argument("--f", type=str, default="test.txt", help="For Jupiter notebook")

args = parser.parse_args()


print(args)


# %%
os.environ["WANDB_API_KEY"] = 'c08ff557ed402c26a0c57ca0e7803529bdba9268'  # This is Leo's API key for the project
wandb.login()
n = 100_000
with wandb.init(entity="leohsieh-epfl", project="szzbpm-distil-5", name='CCC') as run:
# with wandb.init(entity="leohsieh-epfl", project="szzbpm-distil-5", id="", resume='allow') as run:
    wandb.log({"progress": 123})
    print('wandb.log(progress: 123)')
    time.sleep(1)
    for i in tqdm(range(n), disable=not (args.verbose)):
        # time.sleep(1)
        # time.sleep(1/n)
        wandb.log({"progress": i})
    print('Run finish')
    pass
print("With finished!")
wandb.finish()
print("wandb.finish() completed!")

# leohsieh-epfl/szzbpm-distil-5/cjqaq7ui

# wandb sync --id="cjqaq7ui" --entity="leohsieh-epfl" --project="szzbpm-distil-5"




# --id	The run you want to upload to.
# -p, --project	The project you want to upload to.
# -e, --entity

# %%
exit()


def called_by_wandb_sweep_agent():
    with wandb.init(entity="leohsieh-epfl", project="szzbpm-distil-5", name="BBB"):
        for i in tqdm(range(1000), disable=not (args.verbose)):
            print(f"i: {i}")
            wandb.log({"progress": i})
    return
# leohsieh-epfl/szzbpm-distil-5/zwjtsuz1

# # ==== Wandb login and start the sweep agent ====


full_sweep_id = "leohsieh-epfl/szzbpm-distil-5/k490emng"
wandb.agent(full_sweep_id, called_by_wandb_sweep_agent, count=1)
wandb.finish()
print("wandb.finish() completed!")


# https://wandb.ai/leohsieh-epfl/pytorch-demo/runs/q2wuhjfb/overview
