import torch
import torch.nn.functional as F
from torch import nn


class MLP(torch.nn.Module):
    def __init__(self, layer_spec):
        super().__init__()
        self.layer_spec = layer_spec
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(layer_spec[0], layer_spec[1]))
        for i in range(1, len(layer_spec) - 1):
            self.layers.append(torch.nn.Linear(layer_spec[i], layer_spec[i + 1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


class CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.ff1 = nn.Linear(64 * 7 * 7, 512)
        self.ff2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.ff1(x.view(x.size(0), -1)))
        return self.ff2(x)
