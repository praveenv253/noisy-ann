#!/usr/bin/env python3

from types import SimpleNamespace
import models


def init_params():
    params = SimpleNamespace()
    params.batch_size = 64
    params.num_epochs = 5
    params.adam_lr = 0.001

    #params.Net = models.Mnist_6L_CNN
    #params.NoisyNet = models.Noisy_Mnist_6L_CNN
    params.Net = models.Mnist_6L_CNN_v2
    params.NoisyNet = models.Noisy_Mnist_6L_CNN_v2

    #params.net_name = 'mnist-6l-cnn'
    params.net_name = 'mnist-6l-cnn-v2'

    return params
