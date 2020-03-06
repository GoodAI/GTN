from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from badger_utils.sacred import Serializable
from badger_utils.torch.serializable_module import SerializableModule
from torch import Tensor
from badger_utils.torch.device_aware import DeviceAwareModule


class TeacherInputType(Enum):
    RANDOM = 1,
    RANDOM_FIXED = 2,
    LEARNED = 3,

    @staticmethod
    def from_string(value: str) -> 'TeacherInputType':
        if value == 'random':
            return TeacherInputType.RANDOM
        elif value == 'random_fixed':
            return TeacherInputType.RANDOM_FIXED
        elif value == 'learned':
            return TeacherInputType.LEARNED
        else:
            raise ValueError(f'Unrecognized value "{value}"')


@dataclass
class MNISTTeacherConfig:
    input_size: int  # input size
    output_size: int  # width of final image
    fc1_size: int
    fc2_filters: int  # number of "filters" of conv-like FC layer, size is fc2_filters * width * width
    conv1_filters: int
    target_classes: int
    input_count: int
    batch_size: int
    input_type: TeacherInputType


class Teacher(SerializableModule):
    learner_optim_params: nn.Parameter


class MNISTTeacher(Teacher):
    c1_kernel_size: int = 3
    c2_kernel_size: int = 3
    leaky_relu_alpha: float = 0.1
    batch_norm_momentum: float = 0.1
    conv2_filters: int = 1
    config: MNISTTeacherConfig

    def __init__(self, config: MNISTTeacherConfig, device: Optional[str] = None):
        super().__init__(device)
        self.config = config
        fc1_size = 1024
        self.fc2_filters = 128
        self.fc2_width = config.output_size  # // 4  #  output_size should be divided by 4 (as in paper), but then dimensions doesn't fit
        fc2_size = self.fc2_filters * self.fc2_width * self.fc2_width

        self.fc1 = nn.Linear(config.input_size + config.target_classes, fc1_size)
        nn.init.kaiming_normal_(self.fc1.weight, self.leaky_relu_alpha)
        self.bn_fc1 = nn.BatchNorm1d(fc1_size, momentum=self.batch_norm_momentum)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        nn.init.kaiming_normal_(self.fc2.weight, self.leaky_relu_alpha)
        self.bn_fc2 = nn.BatchNorm2d(self.fc2_filters, momentum=self.batch_norm_momentum)  # should be 1D or 2D ?
        self.conv1 = nn.Conv2d(128, config.conv1_filters, self.c1_kernel_size, 1, padding=self.c1_kernel_size // 2)
        self.bn_conv1 = nn.BatchNorm2d(config.conv1_filters, momentum=self.batch_norm_momentum)
        self.conv2 = nn.Conv2d(config.conv1_filters, self.conv2_filters, self.c2_kernel_size, 1,
                               padding=self.c2_kernel_size // 2)
        self.bn_conv2 = nn.BatchNorm2d(self.conv2_filters, momentum=self.batch_norm_momentum)
        self.tanh = nn.Tanh()

        self.learner_optim_params = nn.Parameter(torch.tensor([0.02, 0.5]), True)
        # input_target = torch.randint(config.target_classes, (config.input_count, config.batch_size, 1), device=self.device)
        input_target = torch.tensor([x % config.target_classes for x in range(config.batch_size)],
                                    device=self.device).view(1, config.batch_size, 1).expand(config.input_count, -1, -1)
        input_data = self._create_input_data()
        # self.input_data = nn.Parameter(input_data, True) if self.config.input_type == TeacherInputType.LEARNED else input_data
        self._input_data = nn.Parameter(input_data, True) if self.config.input_type == TeacherInputType.LEARNED else input_data
        self.input_target = input_target

        self.to(self.device)

    def _create_input_data(self) -> Tensor:
        return torch.rand((self.config.input_count, self.config.batch_size, self.config.input_size), device=self.device)

    @property
    def input_data(self) -> Tensor:
        if self.config.input_type == TeacherInputType.RANDOM:
            return self._create_input_data()
        else:
            return self._input_data

    def forward(self, x: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        x = torch.cat([x, target], dim=1)
        x = self.fc1(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = self.bn_fc1(x)

        x = self.fc2(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)
        x = x.view(-1, self.fc2_filters, self.fc2_width, self.fc2_width)
        x = self.bn_fc2(x)

        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)

        x = self.conv2(x)
        x = self.bn_conv2(x)
        x = F.leaky_relu(x, self.leaky_relu_alpha)

        x = self.tanh(x)
        # x = torch.flatten(x, 1)

        return x, target
