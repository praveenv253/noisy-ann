#!/usr/bin/env python3

import os
import joblib

import numpy as np
from scipy import ndimage

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


class MnistData:

    def __init__(self, params, train=True, rotation_angle=0, random=False,
                 traineval=False):

        self.train = train
        self.rotation_angle = rotation_angle
        self.random = random

        if rotation_angle == 0:
            self._load_mnist_data()
        else:
            # Check if required rotation exists
            filename = ('../data/MNIST/rotated/mnist'
                        + ('_train' if train else '_test')
                        + ('_random' if random else '')
                        + ('_%d.pkl' % rotation_angle))
            if os.path.isfile(filename):
                self.dataset = joblib.load(filename)
            else:
                self._load_mnist_data()
                self._rotate_data()
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                joblib.dump(self.dataset, filename)

        # Leave out one digit if leave-one-out has been specified
        if params.args.loo is not None:
            self.dataset = [(image, label) for image, label in self.dataset
                            if label != params.args.loo]

        self.traineval = traineval
        if traineval or not train:
            # Setup for evaluating performance on training or test data
            self.loader = DataLoader(self.dataset, batch_size=len(self.dataset),
                                     shuffle=False)
        else:
            # Setup for training
            self.loader = DataLoader(self.dataset, batch_size=params.batch_size,
                                     shuffle=True)


    def _load_mnist_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.dataset = MNIST(root='../data', train=self.train,
                             download=True, transform=transform)


    def _rotate_data(self):
        """
        Rotate each image in the dataset.

        If random is true, each data point is rotated by a random angle drawn from
        Uniform[-rotation_angle, +rotation_angle].
        """

        rng = np.random.default_rng()

        rotated_images = []
        for i, (image, label) in enumerate(self.dataset):
            if (i + 1) % 10000 == 0:
                print('.', end='', flush=True)
            if self.random:
                angle = rng.uniform(-self.rotation_angle, self.rotation_angle)
            else:
                angle = self.rotation_angle
            rotated_image = ndimage.rotate(image.squeeze().numpy(),
                                           angle=angle, reshape=False)
            rotated_images.append((torch.tensor(rotated_image), label))
        print()

        self.dataset = rotated_images
