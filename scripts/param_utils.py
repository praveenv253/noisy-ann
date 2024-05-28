#!/usr/bin/env python3

from types import SimpleNamespace


def init_params():
    params = SimpleNamespace()
    params.batch_size = 64
    params.num_epochs = 5
    params.adam_lr = 0.001

    return params
