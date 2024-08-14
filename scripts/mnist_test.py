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
        data = load_mnist_data(params)
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
    params = Params(args_needed=['rotate', 'noisy', 'covrot', 'iter'])

    performance = compute_performance(params)
    performance.to_pickle(params.perf_filename())
