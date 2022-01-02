import numpy as np
import os
import sys
import pandas as pd
import argparse
#n_masks, index


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='masks', type=int, default=0)
    parser.add_argument('-i', help='masks', type=int, default=0)
    parser.add_argument('-p', help='path', type=str,
                        default='./runs_no_edges/')
    parser.add_argument('-d', help='dataset', type=str, default='swat')
    args = parser.parse_args()
    print("Gathering for", args.m)
    masks = args.m
    path = args.p
    idx = args.i
    files = os.listdir(path)
    files = list(
        filter(lambda x: '.txt' in x and f'masks_{args.d}_{masks}_{idx}' in x, files))
    results = {'F1 ': [], 'precision ': [], 'recall ': [],
               'AUC ': [],'F1_loss ': [], 'precision_loss ': [], 'recall_loss ': [],
               'AUC_loss ': [], 'test_loss ': [], 'val_loss ': []}
    print(files)
    for f in files:
        with open(os.path.join(path, f), 'r') as f:
            lines = f.readlines()[-10:]
            print(lines)
            for l in lines:
                key, val = l.split(':')
                val = val.strip()
                results[key].append(float(val))

    print(results)
    df = pd.DataFrame(results).T
    print(df.to_csv())
    print(df.mean(axis=1))
