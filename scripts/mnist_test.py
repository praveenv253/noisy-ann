#!/usr/bin/env python3

from types import SimpleNamespace

import argparse
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix

from param_utils import Params
from data_utils import MnistData


def compute_performance(params):
    # Load the network to be evaluated
    net = params.Net(params)
    net.load_state_dict(torch.load(params.model_filename()))

    # Evaluate on various fixed rotations of the test data
    performance = []
    for test_angle in [0, 15, 30, 45, 60, 90]:
        data = MnistData(params, train=False, rotation_angle=test_angle)

        images, labels = next(iter(data.loader))
        total = len(data.dataset)

        with torch.no_grad():
            outputs = net(images.reshape((-1, 1, 28, 28)))
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()

        recall = correct / total

        if params.args.confusion:
            conf_mat = confusion_matrix(labels.numpy(), predicted.numpy(), normalize='true')
            performance.append({'test_angle': test_angle, 'recall': recall,
                                'conf_mat': conf_mat})
        else:
            performance.append({'test_angle': test_angle, 'recall': recall})

    performance = pd.DataFrame.from_records(performance).set_index('test_angle')
    return performance


if __name__ == '__main__':
    params = Params(args_needed=['rotate', 'noisy', 'covrot', 'iter', 'confusion'])

    performance = compute_performance(params)
    performance.to_pickle(params.perf_filename())
