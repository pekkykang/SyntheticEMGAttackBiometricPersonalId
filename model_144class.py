
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal



class Net_144(nn.Module):
    def __init__(self, class_num=144, base_features=64, window_length=128, input_channels=128):
        super(Net_144, self).__init__()
        self.class_num = class_num
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  # for EMG images, the channels is 1. not the signal channels: input_channels
                      out_channels=base_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=base_features,
                      out_channels=base_features,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=base_features,
                      out_channels=base_features,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=base_features,
                      out_channels=base_features,
                      kernel_size=1, stride=1),
            nn.BatchNorm2d(base_features),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fcn1 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.fcn2 = nn.Linear(1024, self.class_num)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.fcn1(x.view(x.size(0), -1))
        x = self.fcn2(x)
        x = F.softmax(x, dim=1)
        return x
