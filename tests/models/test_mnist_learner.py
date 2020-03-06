import torch
import pytest

from gtn.models.mnist_learner import MNISTLearner, MNISTLearnerConfig


class TestMNISTLearner:

    def test_forward(self):
        m = MNISTLearner(MNISTLearnerConfig(32, 64, 64, 10))
        x = m.forward(torch.rand((20, 1, 28, 28), device='cuda'))
        assert [20, 10] == list(x.shape)
