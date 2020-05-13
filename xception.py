import torch.nn as nn
from entryblock import EntryBlock
from middleblock import MiddleBlock
from exitblock import ExitBlock
from separable import SeparableConv2d


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

        self.block4 = MiddleBlock(728, 728, 3)
        self.block5 = MiddleBlock(728, 728, 3)
        self.block6 = MiddleBlock(728, 728, 3)
        self.block7 = MiddleBlock(728, 728, 3)

        self.block8 = MiddleBlock(728, 728, 3)
        self.block9 = MiddleBlock(728, 728, 3)
        self.block10 = MiddleBlock(728, 728, 3)
        self.block11 = MiddleBlock(728, 728, 3)

        self.block12 = ExitBlock(728, 1024, 2)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)

