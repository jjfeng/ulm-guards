"""
aggregate table result_files
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List

from common import *

def parse_args(args):
    ''' parse command line arguments '''

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-files',
        type=str)
    parser.add_argument('--log-file',
        type=str,
        default="_output/log.txt")
    parser.set_defaults()
    args = parser.parse_args()
    args.result_files = process_params(args.result_files, str)
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    logging.basicConfig(format="%(message)s", filename=args.log_file, level=logging.DEBUG)
    all_results = [pickle_from_file(f) for f in args.result_files]
    res = pd.concat(all_results)
    agg_res = res.groupby(['key', 'type']).agg([
        np.mean,
        lambda x: np.std(x)/np.sqrt(len(args.result_files))])
    print(agg_res)
    logging.info(agg_res)


if __name__ == "__main__":
    main(sys.argv[1:])
