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

    The layer at which noise is injected can be specified by the kw arg
    `noisy_layer` during init, and is inferred from params otherwise.

    The shape of the noise injected is specified by `cov`.

    Layers are zero-indexed. The 0-th layer is the first hidden layer (the
    input layer is not counted) and the last layer is the output layer.

    Parameters
    ----------

    `params`: param_utils.Params
        used to get `params.activn`, `params.default_noisy_layer` and
        `params.cov_filename()`.

    `noisy`: False, True, 'zero', 'identity', or 'diagonal'
        If set to False, the noiseless version of the network is instantiated.
        If `noisy` is 'zero', don't add noise, but train only post-noise layers.
        If `noisy` is 'identity', use an identity covariance matrix.
        If `noisy` is True, use the saved covariance matrix; and
        if `noisy` is 'diagonal', set its non-diagonal entries to zero.

    `noisy_layer`: int
        Provides a hook to manually override the default in params.
    """

    def __init__(self, params, noisy=False, noisy_layer=None):
        super().__init__()

        self.activn = F.relu if params.activn == 'relu' else torch.tanh
        self.noisy = noisy

        # Set the layer at which to add noise
        # This value is required even if noisy is False, e.g., when extracting
        # activations of the noisy layer to compute the covariance
        if noisy_layer is None:
            self.noisy_layer = params.default_noisy_layer
        else:
            self.noisy_layer = noisy_layer

        if noisy:
            # Freeze all layers prior to the noisy layer
            self._freeze_layers()

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


    def _freeze_layers(self):
        """
        Freezes layers upto and including `noisy_layer`. When noise is added to
        the hidden neurons of `noisy_layer`, only subsequent layers need
        retraining. Assumes that layer names contain the layer number as the
        last character.
        """
        for name, parameter in self.named_parameters():
            if self._layer_num(name) <= self.noisy_layer:
                parameter.requires_grad = False


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



class Mnist_6L_CNN(NoisyModule):
    """
    6-layer feedforward neural network with one convolutional layer.

    """

    _layer_shapes = [(32, 14, 14), (128,), (64,), (32,), (16,), (10,)]
    # Last layer is the output layer

    def __init__(self, params, noisy=False, noisy_layer=None):
        super().__init__(params, noisy, noisy_layer)

        self.conv0 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*14*14, 128)  # 32 chans, image 14x14 after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)
        self.outputs = []    # Excludes convolutional output before max-pooling


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



class Mnist_6L_CNN_v2(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Image remains 28x28 after this
        # max pool down to 14x14 after this
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # Image remains 14x14
        # No max pooling here
        self.conv3 = nn.Conv2d(16, 16, kernel_size=5, padding=0) # Image becomes 10x10
        # max pool down to 5x5
        self.fc3 = nn.Linear(16*5*5, 32)  # 16 channels, image size 5x5 after max pooling
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)
        self.outputs = []    # Excludes convolutional output before max-pooling
        self.all_outputs = []

        try:
            self.activn = (F.tanh if params.activn == 'tanh' else F.relu)
        except:
            self.activn = F.relu

    def forward(self, x):
        conv1_out = self.activn(self.conv1(x))
        maxpool1_out = F.max_pool2d(conv1_out, kernel_size=2, stride=2)
        conv2_out = self.activn(self.conv2(maxpool1_out))
        conv3_out = self.activn(self.conv3(conv2_out))
        maxpool3_out = F.max_pool2d(conv3_out, kernel_size=2, stride=2)
        fc3_out = self.activn(self.fc3(maxpool3_out.view(-1, 16*5*5)))
        fc4_out = self.activn(self.fc4(fc3_out))
        fc5_out = self.fc5(fc4_out)

        # Save outputs
        self.outputs = [maxpool1_out.view(x.shape[0], -1),
                        conv2_out.view(x.shape[0], -1),
                        maxpool3_out.view(x.shape[0], -1),
                        fc3_out, fc4_out, fc5_out]
        self.all_outputs = [conv1_out, maxpool1_out, conv2_out, conv3_out,
                            maxpool3_out, fc3_out, fc4_out, fc5_out]
        return fc5_out

    def noisy_layer_output(self):
        return self.all_outputs[-3]


class Noisy_Mnist_6L_CNN_v2(nn.Module):
    """
    Class that builds on a pre-trained neural network, and injects noise in
    one layer of the network in its forward pass.

    Trainable parameters are only for the layer after that.

    This network currently retrains the last two layers.
    """

    noise_dim = 32

    def __init__(self, oldnet, cov):
        super().__init__()

        # Run the first few layers of the old network. Placing it in a function
        # ensures that the oldnet doesn't get included in the NoisyNet's params
        def oldfwd(x):
            oldnet(x)
            return oldnet.all_outputs[-3]
        self.oldfwd = oldfwd
        self.activn = oldnet.activn

        #self.oldnet = Net()
        #self.oldnet.load_state_dict(oldnet.state_dict())
        self.cov = cov
        self.rng = np.random.default_rng()
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        with torch.no_grad():
            old_out = self.oldfwd(x)

        noise = self.rng.multivariate_normal(np.zeros(self.cov.shape[0]),
                                             self.cov, size=old_out.shape[0]
                                            ).astype(np.float32)
        fc4_out = self.activn(self.fc4(old_out + torch.tensor(noise)))
        fc5_out = self.fc5(fc4_out)

        #self.outputs = self.oldnet.outputs[:-2] + [fc4_out, fc5_out]
        return fc5_out


class Mnist_5L_CNN_v3(nn.Module):
    def __init__(self, params=None):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Image remains 28x28 after this
        # max pool down to 14x14 after this
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0) # Image becomes 10x10
        # max pool down to 5x5
        self.fc3 = nn.Linear(16*5*5, 32)  # 16 channels, image size 5x5 after max pooling
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)
        self.outputs = []    # Excludes convolutional output before max-pooling
        self.all_outputs = []

        try:
            self.activn = (F.tanh if params.activn == 'tanh' else F.relu)
        except:
            self.activn = F.relu

    def forward(self, x):
        conv1_out = self.activn(self.conv1(x))
        maxpool1_out = F.max_pool2d(conv1_out, kernel_size=2, stride=2)
        conv2_out = self.activn(self.conv2(maxpool1_out))
        maxpool2_out = F.max_pool2d(conv2_out, kernel_size=2, stride=2)
        fc3_out = self.activn(self.fc3(maxpool2_out.view(-1, 16*5*5)))
        fc4_out = self.activn(self.fc4(fc3_out))
        fc5_out = self.fc5(fc4_out)

        # Save outputs
        self.outputs = [maxpool1_out.view(x.shape[0], -1),
                        maxpool2_out.view(x.shape[0], -1),
                        fc3_out, fc4_out, fc5_out]
        self.all_outputs = [conv1_out, maxpool1_out, conv2_out, maxpool2_out,
                            fc3_out, fc4_out, fc5_out]
        return fc5_out

    def noisy_layer_output(self):
        """
        Output of layer to be used to compute covariance.
        """
        shape = self.all_outputs[1].shape
        return self.all_outputs[1].reshape((shape[0], shape[1] * shape[2] * shape[3]))


class Noisy_Mnist_5L_CNN_v3(nn.Module):
    """
    Class that builds on a pre-trained neural network, and injects noise in
    one layer of the network in its forward pass.

    Trainable parameters are only for the layer after that.

    This network currently retrains everything except the first layer.
    """

    noise_dim = 6 * 14 * 14

    def __init__(self, oldnet, cov):
        super().__init__()

        # Run the first few layers of the old network. Placing it in a function
        # ensures that the oldnet doesn't get included in the NoisyNet's params
        def oldfwd(x):
            oldnet(x)
            return oldnet.all_outputs[1]
        self.oldfwd = oldfwd
        self.activn = oldnet.activn

        self.cov = cov
        self.rng = np.random.default_rng()
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0) # Image becomes 10x10
        # max pool down to 5x5
        self.fc3 = nn.Linear(16*5*5, 32)  # 16 channels, image size 5x5 after max pooling
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        with torch.no_grad():
            old_out = self.oldfwd(x)

        noise = self.rng.multivariate_normal(np.zeros(self.cov.shape[0]),
                                             self.cov, size=old_out.shape[0]
                                            ).astype(np.float32).reshape(old_out.shape)
        conv2_out = self.activn(self.conv2(old_out + torch.tensor(noise)))
        maxpool2_out = F.max_pool2d(conv2_out, kernel_size=2, stride=2)
        fc3_out = self.activn(self.fc3(maxpool2_out.view(-1, 16*5*5)))
        fc4_out = self.activn(self.fc4(fc3_out))
        fc5_out = self.fc5(fc4_out)

        #self.outputs = self.oldnet.outputs[:2] + [fc4_out, fc5_out]
        return fc5_out


class Mnist_5L_CNN_v3_1(nn.Module):
    """
    This architecture is identical to Mnist_5L_CNN_v3, but noise is injected
    after the second layer, instead of after the first.
    """

    def __init__(self, params=None):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # Image remains 28x28 after this
        # max pool down to 14x14 after this
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=0) # Image becomes 10x10
        # max pool down to 5x5
        self.fc3 = nn.Linear(16*5*5, 32)  # 16 channels, image size 5x5 after max pooling
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)
        self.outputs = []    # Excludes convolutional output before max-pooling
        self.all_outputs = []

        try:
            self.activn = (F.tanh if params.activn == 'tanh' else F.relu)
        except:
            self.activn = F.relu

    def forward(self, x):
        conv1_out = self.activn(self.conv1(x))
        maxpool1_out = F.max_pool2d(conv1_out, kernel_size=2, stride=2)
        conv2_out = self.activn(self.conv2(maxpool1_out))
        maxpool2_out = F.max_pool2d(conv2_out, kernel_size=2, stride=2)
        fc3_out = self.activn(self.fc3(maxpool2_out.view(-1, 16*5*5)))
        fc4_out = self.activn(self.fc4(fc3_out))
        fc5_out = self.fc5(fc4_out)

        # Save outputs
        self.outputs = [maxpool1_out.view(x.shape[0], -1),
                        maxpool2_out.view(x.shape[0], -1),
                        fc3_out, fc4_out, fc5_out]
        self.all_outputs = [conv1_out, maxpool1_out, conv2_out, maxpool2_out,
                            fc3_out, fc4_out, fc5_out]
        return fc5_out

    def noisy_layer_output(self):
        """
        Output of layer to be used to compute covariance.
        """
        shape = self.all_outputs[3].shape
        return self.all_outputs[3].reshape((shape[0], -1))


class Noisy_Mnist_5L_CNN_v3_1(nn.Module):
    """
    Class that builds on a pre-trained neural network, and injects noise in
    one layer of the network in its forward pass.

    Trainable parameters are only for the layer after that.

    This network currently retrains everything except the first two layers.
    """

    noise_dim = 16 * 5 * 5

    def __init__(self, oldnet, cov):
        super().__init__()

        # Run the first few layers of the old network. Placing it in a function
        # ensures that the oldnet doesn't get included in the NoisyNet's params
        def oldfwd(x):
            oldnet(x)
            return oldnet.all_outputs[3]
        self.oldfwd = oldfwd
        self.activn = oldnet.activn

        self.cov = cov
        self.rng = np.random.default_rng()
        self.fc3 = nn.Linear(16*5*5, 32)  # 16 channels, image size 5x5 after max pooling
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        with torch.no_grad():
            old_out = self.oldfwd(x)

        noise = self.rng.multivariate_normal(np.zeros(self.cov.shape[0]),
                                             self.cov, size=old_out.shape[0]
                                            ).astype(np.float32).reshape(old_out.shape)
        fc3_out = self.activn(self.fc3((old_out + torch.tensor(noise)).view(-1, 16*5*5)))
        fc4_out = self.activn(self.fc4(fc3_out))
        fc5_out = self.fc5(fc4_out)

        return fc5_out
