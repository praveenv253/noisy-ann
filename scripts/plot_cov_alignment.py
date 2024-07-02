#!/usr/bin/env python3

from __future__ import print_function, division

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from param_utils import init_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--covrot', type=float, default=60.0,
                        help='Rotation angle to create a distribution')
    args = parser.parse_args()

    params = init_params()

    savefile_name = ('../saved-models/covariance-alignment--%s--%s--%.2f.csv'
                     % (params.net_name, params.activn, args.covrot))
    df = pd.read_csv(savefile_name)
    num_layers = df['layer'].nunique()

    df_ = df.set_index(['layer', 'digit1', 'digit2'])['alignment']
    for i in sorted(df['layer'].unique()):
        plt.figure()
        sns.heatmap(data=df_.xs(i, level='layer').unstack(), square=True,
                    annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1)
        plt.title('layer = %d' % i)

    plt.figure()
    df_ = df.reset_index().drop(columns=['digit1', 'digit2']).set_index('layer')
    df_.columns.rename('metric', inplace=True)
    df_ = df_.stack().rename('value').reset_index()
    sns.lineplot(data=df.reset_index(), x='layer', y='alignment')
    sns.lineplot(data=df.reset_index(), x='layer', y='pancakeness')

    plt.show()
