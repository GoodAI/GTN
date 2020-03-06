from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from badger_utils.torch.device_aware import DeviceAwareModule

from gtn.models.learner import Learner


@dataclass
class MNISTLearnerConfig:
    conv1_filters: int
    conv2_filters: int
    fc_size: int
    target_classes: int



class MNISTLearner(Learner):
    c1_kernel_size: int = 3
    c2_kernel_size: int = 3
    mp1_size: int = 2
    mp2_size: int = 2
    input_size: int = 28
    leaky_relu_alpha: float = 0.1
    batch_norm_momentum: float = 0.1  # should be 0.0 as written in paper, but it reduces the performance

    def __init__(self, params: MNISTLearnerConfig, device: Optional[str] = None):
        super().__init__(device)
        self.conv1 = nn.Conv2d(1, params.conv1_filters, self.c1_kernel_size, 1)
        self.bn1 = nn.BatchNorm2d(params.conv1_filters, momentum=self.batch_norm_momentum)
        self.conv2 = nn.Conv2d(params.conv1_filters, params.conv2_filters, self.c2_kernel_size, 1)
        self.bn2 = nn.BatchNorm2d(params.conv2_filters, momentum=self.batch_norm_momentum)
        c1_size = (self.input_size - self.c1_kernel_size + 1) // self.mp1_size
        c2_size = (c1_size - self.c2_kernel_size + 1) // self.mp2_size
        self.fc = nn.Linear(params.conv2_filters * c2_size * c2_size, params.target_classes)
        nn.init.kaiming_normal_(self.fc.weight, self.leaky_relu_alpha)
        self.bn_fc = nn.BatchNorm1d(params.target_classes, momentum=self.batch_norm_momentum)
        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.bn1(x)
        x = F.max_pool2d(x, self.mp1_size)

        x = self.conv2(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.bn2(x)
        x = F.max_pool2d(x, self.mp2_size)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.bn_fc(x)

        output = F.log_softmax(x, dim=1)
        return output

    def reset(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.apply(weight_reset)
