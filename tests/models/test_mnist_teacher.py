import torch
import pytest
from badger_utils.torch.test_utils import same

from gtn.models.mnist_teacher import MNISTTeacher, MNISTTeacherConfig, TeacherInputType


class TestMNISTTeacher:

    def test_forward(self):
        bs = 5
        image_size = 28
        m = MNISTTeacher(MNISTTeacherConfig(16, image_size, 1024, 128, 64, 1, 16, bs, TeacherInputType.RANDOM))
        targets = torch.tensor([3.0], device='cuda').expand((bs, 1))
        x, label = m.forward(torch.rand((bs, 16), device='cuda'), targets)
        assert [bs, 1, image_size, image_size] == list(x.shape)
        assert same(targets, label)
