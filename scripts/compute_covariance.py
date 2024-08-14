#!/usr/bin/env python3

import argparse

import numpy as np
import numpy.linalg as la
import torch

from param_utils import Params
from data_utils import load_mnist_data, rotate_images, setup_dataloaders


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--covrot', type=int, default=60,
                        help='Rotation angle to use for computing covariance')
    parser.add_argument('--iter', type=int, default=0)
    args = parser.parse_args()

    params = Params(args)

    # Load the original unrotated dataset
    data = load_mnist_data(params)
    setup_dataloaders(data, params, traineval=True)

    # Load data and create the rotated training dataset
    data_rot = load_mnist_data(params)
    data_rot.trainset = rotate_images(data_rot.trainset, args.covrot,
                                      random=True)
    setup_dataloaders(data_rot, params, traineval=True)

    # Load the network
    net = params.Net(params)
    net.load_state_dict(torch.load(params.vert_model_filename()))

    # Compute covariance matrix on forward pass
    images, _ = next(iter(data.trainloader))
    images_rot, _ = next(iter(data_rot.trainloader))
    with torch.no_grad():
        # Get response to the rotated images
        net(images_rot.reshape((-1, 1, 28, 28)))  # Forward pass
        activn_rot = net.noisy_layer_output().numpy().copy()
        # Create copy to ensure we aren't working with a reference to the tensor

        # Get response to the original images
        net(images.reshape((-1, 1, 28, 28)))
        activn = net.noisy_layer_output().numpy()

    # We are relying on the fact that the rotation function as well as the
    # dataloader (with traineval=True) do not change the ordering of samples
    diff = activn_rot - activn
    cov = np.cov(diff.T)

    print(cov.shape)

    # Check basic statistics of the covariance matrix
    lamda = la.eigvalsh(cov)
    print('Eigenvalues of covariance matrix:')
    print(np.sort(lamda))

    # Save the covariance matrix
    np.save(params.cov_filename(), cov)
