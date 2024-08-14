#!/usr/bin/env python3

import argparse
import itertools as it

import numpy as np
import scipy.linalg as la
import pandas as pd
import torch

from param_utils import Params
from data_utils import load_mnist_data, rotate_images, setup_dataloaders


def compute_alignment(eigvecs_a, cov_b, eigvals_b):
    """
    Compute the alignment of dist_b with respect to dist_a directions.

    dist_a and dist_b should be arrays of size (num_samples, num_features)
    """

    # Compute q-values along the directions of each eigenvector of cov_a
    #q_vals = np.einsum('ij,jk,ki->i', eigvecs_a.T, cov_b, eigvecs_a)
    q_vals = np.array([eigvecs_a[:, i] @ cov_b @ eigvecs_a[:, i]
                       for i in range(cov_b.shape[0])])
    q_vals /= q_vals.sum()  # shape (num_features,)
    eigvals_b_norm = eigvals_b / eigvals_b.sum()

    # Taking mean of the cumulative sum array gives the area under the curve
    q_auc = np.cumsum(q_vals).mean()
    lamda_auc = np.cumsum(eigvals_b_norm).mean()

    # This alignment index is designed to be between -1 and 1, with 0 being
    # the expected alignment with random projections, 1 being fully aligned
    # and -1 being completely mis-aligned, i.e. aligned with lamda[::-1]
    alignment_index = (q_auc - 0.5) / (lamda_auc - 0.5)

    # Similarly, we design a pancake index that indicates how squished the
    # overall representation is - it lies between 0 and 1, with 1 being
    # highly squished (all variance explained by a single dimension), and 0
    # meaning all dimensions provide equal variance (isotropic)
    pancake_index = 2 * (lamda_auc - 0.5)

    return alignment_index, pancake_index


if __name__ == '__main__':
    params = Params(args_needed=['covrot', 'iter'])

    # Load the original unrotated dataset
    data = load_mnist_data(params)
    setup_dataloaders(data, params, traineval=True)

    # Load data and create the rotated training dataset
    data_rot = load_mnist_data(params)
    data_rot.trainset = rotate_images(data_rot.trainset, params.args.covrot,
                                      random=True)
    setup_dataloaders(data_rot, params, traineval=True)

    images, _ = next(iter(data.trainloader))
    images_rot, labels = next(iter(data_rot.trainloader))
    labels_numpy = labels.numpy()
    digits = list(labels.unique().numpy())
    digit_indices = {digit: np.where(labels_numpy == digit)[0]
                     for digit in digits}

    # Create two networks, one to store the original activations and one to
    # store the rotated activations
    net = params.Net(params)
    net_rot = params.Net(params)
    loadfile_path = params.vert_model_filename()
    net.load_state_dict(torch.load(loadfile_path))
    net_rot.load_state_dict(torch.load(loadfile_path))

    # Do a forward pass of the entire dataset
    with torch.no_grad():
        net(images.reshape((-1, 1, 28, 28)))  # Forward pass
        net_rot(images_rot.reshape((-1, 1, 28, 28)))  # Forward pass

    # For each layer, we take every ordered pair of digits and compute the
    # alignment of the covariance of the second w.r.t. the first
    rets = []
    for layer in range(len(net.outputs)):
        print(layer, end=': ', flush=True)
        activations = (net_rot.outputs[layer] - net.outputs[layer])
        # activations now has shape (num_samples, num_features)

        # Pre-compute covariance matrices and eigenvalue decompositions
        covs = {}
        eigvals = {}
        eigvecs = {}
        for digit in digits:
            print(digit, end=' ', flush=True)
            activation = activations[digit_indices[digit]]
            covs[digit] = np.cov(activation.T)
            lamda, v = la.eigh(covs[digit])
            eigvals[digit] = np.maximum(lamda[::-1], 0)
            eigvecs[digit] = v[:, ::-1]

        for digit1 in digits:
            for digit2 in digits:
                print('.', end='', flush=True)
                if digit1 == digit2:
                    ret = np.nan, np.nan
                else:
                    ret = compute_alignment(eigvecs[digit1], covs[digit2],
                                            eigvals[digit2])
                rets.append({'layer': layer, 'digit1': digit1, 'digit2': digit2,
                             'alignment': ret[0], 'pancakeness': ret[1]})
        print()

    alignment_stats = pd.DataFrame.from_records(rets).set_index(['layer', 'digit1', 'digit2'])

    # Save the covariance matrix
    alignment_stats.to_csv(params.alignment_filename())
