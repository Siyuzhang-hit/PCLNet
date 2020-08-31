# coding=utf-8
import torch
import torch.nn as nn


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class Net(nn.Module):
    def __init__(self, outputdim=32):
        super(Net, self).__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(9, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2))
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2))
        self.conv_block_3 = torch.nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc4 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True))
        self.fc5 = nn.Sequential(
            nn.Linear(64, outputdim))

    def forward(self, x, layer=5):
        if layer <= 0:
            return x
        x = self.conv_block_1(x)
        if layer == 1:
            return x
        x = self.conv_block_2(x)
        if layer == 2:
            return x
        x = self.conv_block_3(x)
        if layer == 3:
            return x
        x = x.view(x.size(0), -1)
        x = self.fc4(x)
        if layer == 4:
            return x
        x = self.fc5(x)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, n_label=15):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Sequential()
        self.classifier.add_module('Flatten', Flatten())
        self.classifier.add_module('LiniearClassifier', nn.Linear(32, n_label))
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)


if __name__ == '__main__':
    model = Net()
