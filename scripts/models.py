#!/usr/bin/env python3

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the neural network architecture
class Mnist_6L_CNN(nn.Module):
    def __init__(self):
        super(Mnist_6L_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32*14*14, 128)  # 32 channels, image size 14x14 after max pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        maxpool1_out = F.max_pool2d(conv1_out, kernel_size=2, stride=2)
        fc1_out = F.relu(self.fc1(maxpool1_out.view(-1, 32*14*14)))
        fc2_out = F.relu(self.fc2(fc1_out))
        fc3_out = F.relu(self.fc3(fc2_out))
        fc4_out = F.relu(self.fc4(fc3_out))
        fc5_out = self.fc5(fc4_out)

        # Save outputs
        self.outputs = [conv1_out, maxpool1_out, fc1_out, fc2_out, fc3_out,
                        fc4_out, fc5_out]
        return fc5_out


class Noisy_Mnist_6L_CNN(nn.Module):
    """
    Class that builds on a pre-trained neural network, and injects noise in
    one layer of the network in its forward pass.

    Trainable parameters are only for the layer after that.

    This network currently retrains the last two layers.
    """
    #TODO: Make the layer of choice into a parameter.

    def __init__(self, oldnet, cov):
        super(Noisy_Mnist_6L_CNN, self).__init__()

        # Run the first few layers of the old network. Placing it in a function
        # ensures that the oldnet doesn't get included in the NoisyNet's params
        def oldfwd(x):
            oldnet(x)
            return oldnet.outputs[-3]
        self.oldfwd = oldfwd

        #self.oldnet = Net()
        #self.oldnet.load_state_dict(oldnet.state_dict())
        self.cov = cov
        self.rng = np.random.default_rng()
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 10)

    def forward(self, x):
        with torch.no_grad():
            old_out = self.oldfwd(x)

        noise = self.rng.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov).astype(np.float32)
        fc4_out = F.relu(self.fc4(old_out + torch.tensor(noise)))
        fc5_out = self.fc5(fc4_out)

        #self.outputs = self.oldnet.outputs[:-2] + [fc4_out, fc5_out]
        return fc5_out
