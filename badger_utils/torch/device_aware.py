from abc import ABC
from typing import Optional

import torch
from torch import nn as nn


def default_device() -> str:
    # return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class DeviceAware(ABC):
    device: str = 'cpu'

    def __init__(self, device: Optional[str]):
        if device is None:
            device = default_device()
        self.device = device


class DeviceAwareModule(nn.Module):
    device: str = 'cpu'

    def __init__(self, device: Optional[str]):
        super().__init__()
        if device is None:
            device = default_device()
        self.device = device

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        result.device = args[0]  # hack to extract device
        return result
