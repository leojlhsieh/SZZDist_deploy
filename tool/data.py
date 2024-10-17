import os
import sys
sys.path.append(os.getcwd())
import shutil
from datasets import arrow_dataset


def clean_dir(path: str):
    """Remove all files in the directory."""
    if os.path.exists(path):
        shutil.rmtree(path)

def data_saver(path: str, data: arrow_dataset.Dataset):
    """Save the dataset to path."""
    clean_dir(path)
    os.makedirs(path, exist_ok=True)
    data.save_to_disk(path)