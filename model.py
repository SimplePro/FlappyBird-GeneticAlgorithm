import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        self.weight1 = torch.randn(size=((3, 6)))
        self.bias1 = torch.randn(size=(1, 6))

        self.weight2 = torch.randn(size=((6, 1)))
        self.bias2 = torch.randn(size=(1, 1))

    def forward(self, x):
        # print(x)
        # data: (angle, y_velocity, pipe_x, pipe_y, bird_x, bird_y)

        x = x @ self.weight1 + self.bias1
        x = F.leaky_relu(x, negative_slope=0.2)

        x = x @ self.weight2 + self.bias2
        x = F.sigmoid(x)

        return x

    def mutate(self):
        return (
            self.weight1 + (torch.normal(mean=torch.zeros((3, 6)), std=torch.ones((3, 6))*2)),
            self.bias1 + (torch.normal(mean=torch.zeros((1, 6)), std=torch.ones((1, 6))*2)),
            self.weight2 + (torch.normal(mean=torch.zeros((6, 1)), std=torch.ones((6, 1))*2)),
            self.bias2 + (torch.normal(mean=torch.zeros((1, 1)), std=torch.ones((1, 1))*2))
        )