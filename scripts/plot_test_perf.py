#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    df = pd.DataFrame()
    dfs = {}
    for i in range(1, 2):
        d = pd.read_pickle('../saved/perf--mnist-5l-cnn-v3.1--tanh--noisy-zero--rot-60--%d.pkl' % i)
        dfs[i] = d

    baseline_df = pd.concat(dfs)
    baseline_df = baseline_df.droplevel(0, axis=1)
    baseline_df.index.set_names(['iter', 'cond'], inplace=True)
    baseline_df = baseline_df.reset_index()
    baseline_df = baseline_df.replace({'cond': {0: 'Lower baseline', 60: 'Upper baseline'}}).set_index(['iter', 'cond'])

    dfs = {}
    for i in range(1, 2):
        d = pd.read_pickle('../saved/perf--mnist-5l-cnn-v3.1--tanh--noisy--covrot-60--%d.pkl' % i)
        dfs[i] = d
    result_df = pd.concat(dfs, axis=1).T
    result_df.index.set_names(['iter', 'cond'], inplace=True)
    result_df = result_df.reset_index()
    result_df = result_df.replace({'cond': {'recall': 'Result'}}).set_index(['iter', 'cond'])


    dfs = {}
    for i in range(1, 2):
        d = pd.read_pickle('../saved/perf--mnist-5l-cnn-v3.1--tanh--noisy-diagonal--covrot-60--%d.pkl' % i)
        dfs[i] = d
    control_df = pd.concat(dfs, axis=1).T
    control_df.index.set_names(['iter', 'cond'], inplace=True)
    control_df = control_df.reset_index()
    control_df = control_df.replace({'cond': {'recall': 'Control'}}).set_index(['iter', 'cond'])

    #print(baseline_df)
    #print(result_df)
    #print(control_df)

    full_df = pd.concat([baseline_df, result_df, control_df]).sort_index().stack().rename('Accuracy').reset_index()
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
