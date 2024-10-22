# %%
import numpy as np
import random
import itertools
from pprint import pprint
import wandb
import json


# A function to generate log-uniform distribution
def log_uniform_distribution(start: float, end: float, points: int) -> list[float]:
    a, b, n = start, end, points
    log_uniform_points = np.exp(np.linspace(np.log(a), np.log(b), n))
    return log_uniform_points.tolist()


# A function to create all the combinations of hyperparameters into a list of dictionary. using keyword arguments
def create_hyperparameter_combinations(**kwargs):
    keys, values = zip(*kwargs.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


hyperparameters = {
    'bpm_depth': [3, 4, 6],  # Number of SLM modulation (aka bounce) in BPM
    'bpm_width': [150, 300, 600],  # Width of SLM [pixel], height is same as width
    'Ldist': [3e-3, 6e-3, 12e-3],  # Distance between SLM & mirror [m]
    'lr_bpm': log_uniform_distribution(start=5e-3 * 0.1, end=5e-3 * 10, points=7),  # nPOLO paper use lr=5e-3 for optics
    'lr_class': log_uniform_distribution(start=2.5e-4 * 0.1, end=2.5e-4 * 10, points=7),  # nPOLO paper use lr=2.5e-4 for digital classifier
    'loss_ratio': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # Total loss = loss_ratio * L1 + (1-loss_ratio) * L2
}

combinations = create_hyperparameter_combinations(**hyperparameters)
print("Number of combinations:", len(combinations))
pprint(combinations)


# %%
# Shuffle the combinations
random.shuffle(combinations)
print("Shuffled combinations:")
pprint(combinations)
# Save the list to a JSON file
with open('shuffled_combinations.json', 'w') as json_file:
    json.dump(combinations, json_file, indent=4)  # indent for pretty print
print("List saved to data.json.")


# %%
# Load the list from a JSON file
with open('shuffled_combinations.json', 'r') as json_file:
    combinations = json.load(json_file)
print("List loaded from data.json.")
pprint(combinations)


# %%
# Create a list of value, value is from 0 to len(combinations)
sweep_indices = list(range(len(combinations)))
print(sweep_indices)


# %%
# Create wandb sweep configuration
data_name = 'my_imagenette'  # choices=['my_mnist', 'my_fashion_mnist', 'my_cifar10', 'my_imagenette']
entity_name = 'leohsieh-epfl'
project_name = 'szzbpm-distil-5'

sweep_config = {
    'entity': entity_name,
    'project': project_name,
    'name': f'grid-sweep-{data_name}',
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'test/accuracy'},
    'parameters': {'data_name': {'value': data_name},
                   'sweep_index': {'values': sweep_indices},
                   }
}
pprint(sweep_config, sort_dicts=False)


# %%
# Create A wanb Sweep
sweep_id = wandb.sweep(sweep_config)
print(f"Sweep ID: {sweep_id}")

# %%
