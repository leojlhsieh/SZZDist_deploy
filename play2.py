# %%
from argparse import ArgumentParser
import time
import logging

parser = ArgumentParser()
parser.add_argument('--a', type=str, default='world')
parser.add_argument('--b', type=int, default=0)
args = parser.parse_args()

for _ in range(10):
    print(f'print Hello {args.a} {args.b}')
    logging.info(f'log info Hello {args.a} {args.b}')
    time.sleep(1)


