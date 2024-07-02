#!/usr/bin/env python3

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the neural network architecture
class Mnist_6L_CNN(nn.Module):
    def __init__(self, params=None):
        super(Mnist_6L_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*14*14, 128)  # 32 channels, image size 14x14 after max pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
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
        fc1_out = self.activn(self.fc1(maxpool1_out.view(-1, 32*14*14)))
        fc2_out = self.activn(self.fc2(fc1_out))
        fc3_out = self.activn(self.fc3(fc2_out))
        fc4_out = self.activn(self.fc4(fc3_out))
        fc5_out = self.fc5(fc4_out)

        # Save outputs
        self.outputs = [maxpool1_out.view(x.shape[0], -1), fc1_out, fc2_out,
                        fc3_out, fc4_out, fc5_out]
        self.all_outputs = [conv1_out, maxpool1_out, fc1_out, fc2_out, fc3_out,
                            fc4_out, fc5_out]
        return fc5_out

    def noisy_layer_output(self):
        return self.all_outputs[-3]


class Noisy_Mnist_6L_CNN(nn.Module):
    """
    Class that builds on a pre-trained neural network, and injects noise in
    one layer of the network in its forward pass.

    Trainable parameters are only for the layer after that.

    This network currently retrains the last two layers.
    """

    noise_dim = 32

    def __init__(self, oldnet, cov):
        super(Noisy_Mnist_6L_CNN, self).__init__()

        # Run the first few layers of the old network. Placing it in a function
        # ensures that the oldnet doesn't get included in the NoisyNet's params
        def oldfwd(x):
            oldnet(x)
            return oldnet.all_outputs[-3]
        self.oldfwd = oldfwd
        self.activn = oldnet.activn

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


class Mnist_6L_CNN_v2(nn.Module):
    def __init__(self, params=None):
        super(Mnist_6L_CNN_v2, self).__init__()

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
        super(Noisy_Mnist_6L_CNN_v2, self).__init__()

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
        super(Mnist_5L_CNN_v3, self).__init__()

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
        super(Noisy_Mnist_5L_CNN_v3, self).__init__()

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
        super(Mnist_5L_CNN_v3_1, self).__init__()

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
        super(Noisy_Mnist_5L_CNN_v3_1, self).__init__()

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
