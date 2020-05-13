import torch.nn as nn
from separable import SeparableConv2d


class EntryBlock(nn.module):
    def __init__(self, in_filters, out_filters, reps, strides=1, first_relu=True, grow_first=True):
        super(EntryBlock, self).__init__()
        self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
        self.skipbn = nn.BatchNorm2d(out_filters)

        self.relu = nn.ReLU(inplace=True)

        layers = list()
        if first_relu:
            layers.append(self.relu)
        layers.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(out_filters))
        filters = out_filters

        for i in range(reps - 1):
            layers.append(self.relu)
            layers.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(filters))

        if strides > 1:
            layers.append(nn.MaxPool2d(3, strides, 1))
