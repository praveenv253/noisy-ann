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
    for i in list(range(7, 8)):  # Iterate over --iter values
        params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                        args_list=['--iter=%d' % i])
        lower_baseline = pd.read_pickle(params.perf_filename())
        params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                        args_list=['--noisy=zero', '--rotate=60', '--iter=%d' % i, '--reinit'])
        upper_baseline = pd.read_pickle(params.perf_filename())
        params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                        args_list=['--noisy', '--iter=%d' % i, '--reinit'])
        result = pd.read_pickle(params.perf_filename())
        params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                        args_list=['--noisy=diagonal', '--iter=%d' % i, '--reinit'])
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

    sns.set_context('notebook', font_scale=1.25)
    sns.pointplot(data=full_df, x='test_angle', y='Accuracy', hue='cond', errorbar='sd',
                  hue_order=['Lower baseline', 'Upper baseline', 'Control', 'Result'],
                  palette=['grey', 'k', 'C1', 'C0'], linestyles=['--', '--', '-', '-'])
    plt.xlabel('Test rotation')
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[:], labels=labels[:])
    plt.tight_layout()
    plt.show()
