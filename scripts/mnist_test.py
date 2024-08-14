#!/usr/bin/env python3

from types import SimpleNamespace

import argparse
import numpy as np
import torch
import pandas as pd
#from sklearn.metrics import confusion_matrix

from param_utils import Params
from data_utils import load_mnist_data, rotate_images, setup_dataloaders


def compute_performance(params):
    data = load_mnist_data(params)

    # Load the network to be evaluated
    if params.args.noisy:
        oldnet = params.Net(params)
        oldnet.load_state_dict(torch.load(params.vert_model_filename()))
        cov = 0 * np.eye(params.NoisyNet.noise_dim)
        net = params.NoisyNet(oldnet, cov)
    else:
        net = params.Net(params)
    net.load_state_dict(torch.load(params.model_filename()))

    # Evaluate on various fixed rotations of the test data
    performance = []
    for test_angle in [0, 15, 30, 45, 60, 90]:
        if test_angle > 0:
            data.testset = rotate_images(data.testset, test_angle)
        setup_dataloaders(data, params)

        images, labels = next(iter(data.testloader))
        total = len(data.testset)

        with torch.no_grad():
            outputs = net(images.reshape((-1, 1, 28, 28)))
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

        recall = correct / total
        performance.append({'test_angle': test_angle, 'recall': recall})
        #conf_mat = confusion_matrix(labels.numpy(), predicted.numpy())
        #performance.append({'test_angle': test_angle, 'recall': recall,
        #                    'conf_mat': conf_mat})

    performance = pd.DataFrame.from_records(performance).set_index('test_angle')
    return performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate', default=None)
    parser.add_argument('--noisy', nargs='?', const=True, default=False)
    parser.add_argument('--covrot', type=float, default=60.0)
    parser.add_argument('--iter', type=int, default=0)
    args = parser.parse_args()

    if args.rotate == 'all':
        perfs = {}
        #for train_angle in [0, 45, 60]:
        for train_angle in [0, 60]:
            if train_angle:
                args.rotate = train_angle
                args.noisy = 'zero'
            else:
                args.rotate = None
                args.noisy = False
            params = Params(args)
            perfs[train_angle] = compute_performance(params)
        performance = pd.concat(perfs).unstack()
        performance.index.set_names('train_angle', inplace=True)
    else:
        if args.rotate:
            args.rotate = float(args.rotate)
        params = Params(args)
        performance = compute_performance(params)

    #print(performance)
    performance.to_pickle(params.perf_filename())
