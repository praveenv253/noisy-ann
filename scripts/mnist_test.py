#!/usr/bin/env python3

from types import SimpleNamespace

import argparse
import numpy as np
import torch
import pandas as pd

from param_utils import init_params
from data_utils import load_mnist_data, rotate_images, setup_dataloaders


def compute_performance(args):
    params = init_params()

    data = load_mnist_data(params)

    # Load the network to be evaluated
    savefile_name = params.net_name
    if args.noisy:
        savefile_name += '--noisy' + ('-' + args.noisy if args.noisy is not True else '')
        if args.noisy != 'zero':
            savefile_name += '--covrot-%.2f' % args.covrot
    if not args.noisy or args.noisy == 'zero':
        savefile_name += ('--vert' if args.rotate is None else
                          '--rot-%.2f' % args.rotate)
    savefile_name += '.pth'

    if args.noisy:
        oldnet = params.Net()
        oldnet.load_state_dict(torch.load('../saved-models/%s--vert.pth'
                                          % params.net_name))
        cov = np.zeros((32, 32))
        net = params.NoisyNet(oldnet, cov)
    else:
        net = params.Net()
    net.load_state_dict(torch.load('../saved-models/%s' % savefile_name))

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

    performance = pd.DataFrame.from_records(performance).set_index('test_angle')
    return performance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate', default=None)
    parser.add_argument('--noisy', nargs='?', const=True, default=False)
    parser.add_argument('--covrot', type=float, default=60.0)
    args = parser.parse_args()

    if args.rotate == 'all':
        perfs = {}
        for train_angle in [0, 45, 60]:
            if train_angle:
                args.rotate = train_angle
                args.noisy = 'zero'
            else:
                args.rotate = None
                args.noisy = False
            perfs[train_angle] = compute_performance(args)
        performance = pd.concat(perfs).unstack()
        performance.index.set_names('train_angle', inplace=True)
    else:
        if args.rotate:
            args.rotate = float(args.rotate)
        performance = compute_performance(args)

    print(performance)
