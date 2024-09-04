#!/usr/bin/env python3

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal


class NoisyModule(nn.Module):
    """
    Base class to collect some functions that will be used by all noisy ANN
    implementations.

    Can have noise injected at some intermediate layer when `noisy` is True.
    The shape of the noise injected is specified by `cov`.

    The layer at which noise is injected is inferred from params.
    Layers are zero-indexed. The 0-th layer is the first hidden layer (the
    input layer is not counted) and the last layer is the output layer.

    Parameters
    ----------

    `params`: param_utils.Params
        used to get `params.activn`, `params.noisy_layer` and
        `params.cov_filename()`.

    `noisy`: False, True, 'zero', 'identity', or 'diagonal'
        If set to False, the noiseless version of the network is instantiated.
        If `noisy` is 'zero', don't add noise, but train only post-noise layers.
        If `noisy` is 'identity', use an identity covariance matrix.
        If `noisy` is True, use the saved covariance matrix; and
        if `noisy` is 'diagonal', set its non-diagonal entries to zero.
    """

    def __init__(self, params, noisy=False):
        # `noisy` is set separately from params because different files will
        # initialize the network differently.
        super().__init__()

        self.activn = F.relu if params.activn == 'relu' else torch.tanh
        self.noisy = noisy

        # Set the layer at which to add noise
        # This value is required even if noisy is False, e.g., when extracting
        # activations of the noisy layer to compute the covariance
        self.noisy_layer = params.noisy_layer

        # List to collect outputs from different layers - will typically
        # exclude the output of convolutional layers before max-pooling
        self.outputs = []

        if noisy:
            # Initialize the covariance matrix
            if noisy == 'identity':
                self.cov = np.eye(self._noise_dim())
            elif noisy == 'zero':
                self.cov = np.zeros((self._noise_dim(), self._noise_dim()))
            else:
                self.cov = np.load(params.cov_filename())
                assert self._noise_dim() == self.cov.shape[0]
            if noisy == 'diagonal':
                self.cov *= np.eye(self.cov.shape[0])

            # Initialize a random number generator
            self.rng = np.random.default_rng()


    @staticmethod
    def _layer_num(name):
        """Extract layer number from a layer name."""
        return int(name.split('.')[0][-1])


    def _noise_dim(self):
        """Return the size of the noisy layer."""
        return np.prod(self._layer_shapes[self.noisy_layer])


    def _add_noise(self, layer, x):
        """
        Add noise to the activations of the noisy layer.
        """
        if not self.noisy or layer != self.noisy_layer:
            return x

        # Torch does not allow zero-covariance matrices, so this has to be
        # done in numpy, or we need a different mechanism
        #distr = MultivariateNormal(torch.zeros(self.cov.shape[0]),
        #                           covariance_matrix=self.cov)
        #noise = distr.sample(x.shape[0])
        noise = self.rng.multivariate_normal(np.zeros(self.cov.shape[0]),
                                             self.cov, size=x.shape[0])
        noise = torch.tensor(noise.astype(np.float32))
        return x + noise.view(x.shape)


    def noisy_layer_output(self):
        """Return the activations of the noisy layer after a forward pass."""
        try:
            return self.outputs[self.noisy_layer]
        except:
            raise ValueError('Must run a forward pass before calling '
                             'noisy_layer_output')

    def freeze_layers(self):
        """
        Freezes layers upto and including `noisy_layer`. When noise is added to
        the hidden neurons of `noisy_layer`, only subsequent layers need
        retraining. Assumes that layer names contain the layer number as the
        last character.
        """
        for name, parameter in self.named_parameters():
            if self._layer_num(name) <= self.noisy_layer:
                parameter.requires_grad = False


    def post_noise_reinit(self):
        """Reset the weights of the post-noise layers."""
        for name, child in self.named_children():
            if self._layer_num(name) > self.noisy_layer:
                child.reset_parameters()



class Mnist_v1_1C5F(NoisyModule):
    """
    6-layer feedforward neural network with one convolutional layer.
    """

    _layer_shapes = [(32, 14, 14), (128,), (64,), (32,), (16,), (10,)]
    # Last layer is the output layer

    def __init__(self, params, noisy=False):
        super().__init__(params, noisy)

        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*14*14, 128)  # 32 chans, image 14x14 after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)


    def forward(self, x):
        conv0_out = self.activn(self.conv0(x))
        maxpool0_out = self._add_noise(0, F.max_pool2d(conv0_out, kernel_size=2,
                                                       stride=2)
                                      ).view(x.shape[0], -1)
        fc1_out = self._add_noise(1, self.activn(self.fc1(maxpool0_out)))
        fc2_out = self._add_noise(2, self.activn(self.fc2(fc1_out)))
        fc3_out = self._add_noise(3, self.activn(self.fc3(fc2_out)))
        fc4_out = self._add_noise(4, self.activn(self.fc4(fc3_out)))
        fc5_out = self.fc5(fc4_out)

        # Save outputs
        self.outputs = [maxpool0_out, fc1_out, fc2_out, fc3_out, fc4_out, fc5_out]
        return fc5_out



class Mnist_v2_3C3F(NoisyModule):
    """
    6-layer feedforward neural network with three convolutional layers.
    """

    _layer_shapes = [(6, 14, 14), (16, 14, 14), (16, 5, 5), (32,), (16,), (10,)]

    def __init__(self, params, noisy=False):
        super().__init__(params, noisy)

        self.conv0 = nn.Conv2d(1, 6, kernel_size=5, padding=2)   # Image remains 28x28 after this
        # max pool down to 14x14 after this
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # Image remains 14x14
        # No max pooling here
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=0) # Image becomes 10x10
        # max pool down to 5x5
        self.fc3 = nn.Linear(16*5*5, 32)  # 16 channels, image size 5x5 after max pooling
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)


    def forward(self, x):
        conv0_out = self.activn(self.conv0(x))
        maxpool0_out = self._add_noise(0, F.max_pool2d(conv0_out, kernel_size=2,
                                                       stride=2))
        conv1_out = self._add_noise(1, self.activn(self.conv1(maxpool0_out)))
        conv2_out = self.activn(self.conv2(conv1_out))
        maxpool2_out = self._add_noise(2, F.max_pool2d(conv2_out, kernel_size=2,
                                                       stride=2)
                                      ).view(x.shape[0], -1)
        fc3_out = self._add_noise(3, self.activn(self.fc3(maxpool2_out)))
        fc4_out = self._add_noise(4, self.activn(self.fc4(fc3_out)))
        fc5_out = self.fc5(fc4_out)

        # Save outputs
        self.outputs = [maxpool0_out.view(x.shape[0], -1),
                        conv1_out.view(x.shape[0], -1),
                        maxpool2_out, fc3_out, fc4_out, fc5_out]
        return fc5_out



class Mnist_v3_2C3F(NoisyModule):
    """
    5-layer feedforward neural network with two convolutional layers.
    """

    _layer_shapes = [(6, 14, 14), (16, 5, 5), (32,), (16,), (10,)]

    def __init__(self, params, noisy=False):
        super().__init__(params, noisy)

        self.conv0 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Image remains 28x28 after this
        # max pool down to 14x14 after this
        self.conv1 = nn.Conv2d(6, 16, kernel_size=5, padding=0) # Image becomes 10x10
        # max pool down to 5x5
        self.fc2 = nn.Linear(16*5*5, 32)  # 16 channels, image size 5x5 after max pooling
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 10)


    def forward(self, x):
        conv0_out = self.activn(self.conv0(x))
        maxpool0_out = self._add_noise(0, F.max_pool2d(conv0_out, kernel_size=2,
                                                       stride=2))
        conv1_out = self.activn(self.conv1(maxpool0_out))
        maxpool1_out = self._add_noise(1, F.max_pool2d(conv1_out, kernel_size=2,
                                                       stride=2)
                                      ).view(x.shape[0], -1)
        fc2_out = self._add_noise(2, self.activn(self.fc2(maxpool1_out)))
        fc3_out = self._add_noise(3, self.activn(self.fc3(fc2_out)))
        fc4_out = self.fc4(fc3_out)

        # Save outputs
        self.outputs = [maxpool0_out.view(x.shape[0], -1), maxpool1_out,
                        fc2_out, fc3_out, fc4_out]
        return fc4_out
