import torch.nn as nn
from separable import SeparableConv2d


class ExitBlock(nn.module):
    def __init__(self, in_filters, out_filters, strides=1):
        super(ExitBlock, self).__init__()
        self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
        self.skipbn = nn.BatchNorm2d(out_filters)

        self.relu = nn.ReLU(inplace=True)

        layers = list()

        layers.append(self.relu)
        layers.append(SeparableConv2d(in_filters, in_filters, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(in_filters))

        layers.append(self.relu)
        layers.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_filters))
        layers.append(nn.MaxPool2d(3, strides, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, input_mat):
        x = self.rep(input_mat)
        skip = self.skip(input_mat)
        skip = self.skipbn(skip)
        x += skip

        return x
