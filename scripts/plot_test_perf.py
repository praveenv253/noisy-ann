#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from param_utils import Params


if __name__ == '__main__':
    df = pd.DataFrame()
    dfs = {}

    noisy_layer = 1

    for i in list(range(1, 11)):  # Iterate over --iter values
        params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                        args_list=['--iter=%d' % i])
        lower_baseline = pd.read_pickle(params.perf_filename())
        params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                        args_list=['--noisy=zero', '--rotate=60', '--iter=%d' % i,
                                   '--reinit', '--noisy-layer=%d' % noisy_layer])
        upper_baseline = pd.read_pickle(params.perf_filename())
        params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                        args_list=['--noisy', '--iter=%d' % i, '--reinit',
                                   '--noisy-layer=%d' % noisy_layer])
        result = pd.read_pickle(params.perf_filename())
        params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                        args_list=['--noisy=diagonal', '--iter=%d' % i, '--reinit',
                                   '--noisy-layer=%d' % noisy_layer])
        control = pd.read_pickle(params.perf_filename())
        df = pd.concat({'Lower baseline': lower_baseline,
                        'Upper baseline': upper_baseline,
                        'Result': result, 'Control': control}, axis=1)
        dfs[i] = df.droplevel(1, axis=1)

    dfs = pd.concat(dfs)
    dfs.index.set_names('iter', level=0, inplace=True)
    full_df = pd.melt(dfs, var_name='cond', value_name='Accuracy',
                      ignore_index=False).reset_index()

    print(full_df)

    plt.figure(figsize=(6, 5))
    sns.set_context('notebook', font_scale=1.5)
    sns.pointplot(data=full_df, x='test_angle', y='Accuracy', hue='cond', errorbar='sd',
                  hue_order=['Lower baseline', 'Upper baseline', 'Control', 'Result'],
                  palette=['grey', 'k', 'C1', 'C0'], linestyles=['--', '--', '-', '-'])
    #plt.title('Acc on rotated MNIST, noise in layer %d' % noisy_layer, fontsize=20)
    plt.title('Accuracy on rotated MNIST images', fontsize=20)
    plt.xlabel('Rotation angle (degrees)')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    legend_order = [0, 1, 3, 2]
    ax.legend(handles=list(np.array(handles)[legend_order]), frameon=False, loc='lower left',
              labels=list(np.array(['No noise,\nvertical', 'Supervised,\nrotated',
                               'Misaligned noise,\nvertical',
                               'Aligned noise,\nvertical'])[legend_order]))
    plt.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.show()
