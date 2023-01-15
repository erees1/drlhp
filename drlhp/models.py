import torch


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
