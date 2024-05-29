import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type = str,
        help = 'yaml config file')

    args = parser.parse_args()
    return args