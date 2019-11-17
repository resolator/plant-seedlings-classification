#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd


def get_args():
    """Arguments parser."""
    parser = argparse.ArgumentParser(
        description='Script for making two .csv files (train and valid) '
                    'from dataset dir.'
    )
    parser.add_argument('--root-dir', required=True,
                        help='Path to dir with dirs of classes.')
    parser.add_argument('--save-to', required=True,
                        help='Path to dir for save resulted files.')
    parser.add_argument('--ts', '--train-size', type=float, default=0.7,
                        help='Probability of getting a sample in the train '
                             'subset.')

    args = parser.parse_args()
    if (args.ts < 0.0) or (args.ts > 1.0):
        print('Wrong value for "--train-size" (must be between 0.0 and 1.0).')
        exit(1)

    if not os.path.exists(args.save_to):
        try:
            os.mkdir(args.save_to)
            print('\n"save-to" dir has been created.')
        except Exception as e:
            print('"save-to" does not exist and can\'t be created:', e)
            exit(1)

    return args


def main():
    args = get_args()
    train_data = {'paths': [], 'class_names': []}
    valid_data = {'paths': [], 'class_names': []}

    # splitting dataset
    classes = os.listdir(args.root_dir)
    for sub_dir in classes:
        sub_path = os.path.join(args.root_dir, sub_dir)
        for img_name in os.listdir(sub_path):
            img_path = os.path.join(sub_path, img_name)
            if np.random.uniform(0.0, 1.0) < args.ts:
                train_data['paths'].append(img_path)
                train_data['class_names'].append(sub_dir)
            else:
                valid_data['paths'].append(img_path)
                valid_data['class_names'].append(sub_dir)

    train_df = pd.DataFrame(data=train_data)
    valid_df = pd.DataFrame(data=valid_data)

    train_df.to_csv(os.path.join(args.save_to, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(args.save_to, 'valid.csv'), index=False)
    print('Done!')


if __name__ == '__main__':
    main()
