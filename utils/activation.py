import torch
import torch.nn as nn


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(torch.sigmoid(x))
        else:
            return x * torch.sigmoid(x)


class SignedSqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sqrt(torch.relu(x)) - torch.sqrt(torch.relu(-x))



act_dict = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "swish": SWISH,
    "signed_sqrt": SignedSqrt,
    "identity": nn.Identity
}
