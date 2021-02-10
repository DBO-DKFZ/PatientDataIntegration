import os
import sys
import time
import glob
from argparse import ArgumentParser
import subprocess


def main(hparams):
    base = [sys.executable, 'main.py', '--cfg'] 
    cfgs = glob.glob('cfgs_v0/*.yaml')

    for cfg in cfgs:
        print(cfg)
        for i in range(hparams.N):
            print('overall runs : ',hparams.N)
            out = subprocess.call(base + [cfg])
    print('DONE!')

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--N', type=int, default=1)
    hparams = parser.parse_args()
    main(hparams)