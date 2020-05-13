import torch.nn as nn
from entryblock import EntryBlock
from middleblock import MiddleBlock

class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = EntryBlock(64, 128, 2, 2, first_relu=False)
        self.block2 = EntryBlock(128, 256, 2, 2, first_relu=True)
        self.block3 = EntryBlock(256, 728, 2, 2, first_relu=True)
