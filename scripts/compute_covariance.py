#!/usr/bin/env python3

import argparse

import numpy as np
import numpy.linalg as la
import torch

from param_utils import Params
from data_utils import MnistData


if __name__ == '__main__':
    params = Params(args_needed=['covrot', 'iter'])

    # Load the original unrotated dataset
    data = MnistData(params, traineval=True)

    # Load data and create the rotated training dataset
    data_rot = MnistData(params, rotation_angle=params.args.covrot, random=True,
                         traineval=True)

    # Load the network
    net = params.Net(params)
    net.load_state_dict(torch.load(params.vert_model_filename()))

    # Compute covariance matrix on forward pass
    images, _ = next(iter(data.loader))
    images_rot, _ = next(iter(data_rot.loader))
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
