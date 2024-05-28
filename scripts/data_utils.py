#!/usr/bin/env python3

from types import SimpleNamespace
import numpy as np
from scipy import ndimage

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def load_mnist_data(params):
    data = SimpleNamespace()

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    data.trainset = MNIST(root='../data', train=True, download=True,
                          transform=transform)
    data.testset = MNIST(root='../data', train=False, download=True,
                         transform=transform)

    return data


def setup_dataloaders(data, params, traineval=False):
    if traineval:
        # Setup for evaluating performance on training data
        data.trainloader = DataLoader(data.trainset, batch_size=len(data.trainset),
                                      shuffle=False)
    else:
        data.trainloader = DataLoader(data.trainset, batch_size=params.batch_size,
                                      shuffle=True)
    data.testloader = DataLoader(data.testset, batch_size=len(data.testset),
                                 shuffle=False)
    return data


def rotate_images(dataset, rotation_angle, random=False):
    """
    Rotate each image in the dataset.

    If random is true, each data point is rotated by a random angle drawn from
    Uniform[-rotation_angle, +rotation_angle].
    """

    rng = np.random.default_rng()

    rotated_images = []
    for i, (image, label) in enumerate(dataset):
        if (i + 1) % 10000 == 0:
            print('.', end='', flush=True)
        if random:
            angle = rng.uniform(-rotation_angle, rotation_angle)
        else:
            angle = rotation_angle
        rotated_image = ndimage.rotate(image.squeeze().numpy(),
                                       angle=angle, reshape=False)
        rotated_images.append((torch.tensor(rotated_image), label))
    print()

    return rotated_images
