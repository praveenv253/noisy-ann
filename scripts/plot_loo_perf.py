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

    i = 1

    params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                    args_list=['--iter=%d' % i])
    lower_baseline = pd.read_pickle(params.perf_filename())
    params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                    args_list=['--noisy=zero', '--rotate=60', '--iter=%d' % i,
                               '--reinit'])
    upper_baseline = pd.read_pickle(params.perf_filename())
    params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter'],
                    args_list=['--noisy', '--iter=%d' % i, '--reinit' ])
    main = pd.read_pickle(params.perf_filename())

    results = {}
    conf_mats = {}
    for loo in list(range(10)):
        params = Params(args_needed=['noisy', 'rotate', 'covrot', 'iter', 'confusion'],
                        args_list=['--noisy', '--iter=%d' % i, '--reinit',
                                   '--cov-loo=%d' % loo, '--confusion'])
        result_df = pd.read_pickle(params.perf_filename())
        results['LOO %d' % loo] = result_df[['recall']]
        conf_mats['LOO %d' % loo] = result_df[['conf_mat']]

    results |= {'Lower baseline': lower_baseline, 'Upper baseline': upper_baseline,
                'Main Result': main}

    df = pd.concat(results, axis=1).droplevel(1, axis=1)
    conf_mats = pd.concat(conf_mats, axis=1).droplevel(1, axis=1)

    print(df)
    print(conf_mats)

    old_df = df

    sns.set_context('notebook', font_scale=1.25)

    full_df = pd.melt(df, var_name='cond', value_name='Accuracy',
                      ignore_index=False).reset_index()
    ax = sns.pointplot(data=full_df, x='test_angle', y='Accuracy', hue='cond', errorbar='sd',
                       hue_order=['Lower baseline', 'Upper baseline',
                                  *['LOO %d' % i for i in range(10)], 'Main Result'],
                       palette=['grey', 'grey', *['C%d' % i for i in range(10)], 'k'],
                       linestyles=['--', '--', *['-' for i in range(10)], '--'])
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.xlabel('Test rotation')
    plt.ylabel('Recall')
    plt.tight_layout()
    #plt.title('Acc on rotated MNIST, noise in layer %d' % noisy_layer, fontsize=20)
    #ax = plt.gca()
    #handles, labels = ax.get_legend_handles_labels()
    #ax.legend(handles=handles[:], labels=['No noise, vertical', 'Supervised, rotated',
    #                                      'Misaligned noise, vertical', 'Aligned noise, vertical'])

    df = []
    for test_angle in conf_mats.index:
        row = conf_mats.loc[test_angle]
        for loo in range(10):
            mat = row['LOO %d' % loo]
            acc = mat[loo, loo]

            other_mats = row[row.index != 'LOO %d' % loo]
            other_mats_mean = other_mats.mean()
            base_acc = other_mats_mean[loo, loo]

            df.append({'test_angle': test_angle, 'loo': loo, 'loo_acc': acc,
                       'base_acc': base_acc, 'LB': old_df.loc[test_angle, 'Lower baseline']})

    df = pd.DataFrame.from_records(df)
    df = pd.melt(df, id_vars=['test_angle', 'loo'], value_name='acc', var_name='cond').reset_index()
    sns.catplot(kind='point', data=df, col='loo', col_wrap=5, x='test_angle', y='acc', hue='cond')

    plt.show()
