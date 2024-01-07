import torch
import torch.nn.functional as F
from torch import nn


class MLP(torch.nn.Module):
    def __init__(self, layer_spec: list[int]):
        super().__init__()
        self.layer_spec = layer_spec
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(layer_spec[0], layer_spec[1]))
        for i in range(1, len(layer_spec) - 1):
            self.layers.append(torch.nn.Linear(layer_spec[i], layer_spec[i + 1]))

    def forward(self, x: torch.Tensor):  # type: ignore
        for layer in self.layers[:-1]:  # type: ignore
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x


class AtariPolicy(nn.Module):
    """Architecture infered from "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)"""

    def __init__(self, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.ff1 = nn.Linear(32 * 9 * 9, 256)
        self.ff2 = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor):  # type: ignore
        # x: (batch, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.ff1(x.view(x.size(0), -1)))
        return self.ff2(x)


class RoboticsRewardPredictor(nn.Module):
    """Architecture infered from "Deep Reinforcement Learning from Human Feedback" (Christiano et al., 2017)"""

    pass
