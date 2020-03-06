import torch
from torch import Tensor

DEFAULT_EPS = 1e-3


def same(expected: Tensor, value: Tensor, eps: float = DEFAULT_EPS) -> bool:
    return torch.allclose(expected, value, atol=eps)
