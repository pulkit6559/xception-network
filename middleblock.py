import torch.nn as nn
from separable import SeparableConv2d


class MiddleBlock(nn.module):
    def __init__(self, in_filters, out_filters, reps, strides=1):
        super(MiddleBlock, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        layers = list()

        for i in range(reps):
            layers.append(self.relu)
            layers.append(SeparableConv2d(in_filters, in_filters, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(in_filters))

        self.layers = nn.Sequential(*layers)

    def forward(self, input_mat):
        x = self.rep(input_mat)
        x += input_mat
        return x
