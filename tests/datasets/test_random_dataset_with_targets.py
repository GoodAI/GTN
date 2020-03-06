import pytest
import torch

from gtn.datasets.datasets import Datasets
from gtn.datasets.random_dataset import RandomDatasetWithTargets


class TestRandomDatasetWithTargets:
    def test_dataset(self):
        ds = RandomDatasetWithTargets(16, 3, 2, 'cpu')
        assert 3 == len(ds)
        shapes = [[list(i.shape) for i in sample] for sample in ds]
        assert [[16], [1]] == shapes[0]
        assert [[16], [1]] == shapes[1]
        assert [[16], [1]] == shapes[2]

    def test_dataset_out_of_range(self):
        ds = RandomDatasetWithTargets(16, 3, 2, 'cpu')
        with pytest.raises(IndexError):
            _ = ds[3]

    def test_dataloader(self):
        dl = Datasets.random_dataloader_with_targets(4, 8, 9, 2)
        shapes = [[list(i.shape) for i in sample] for sample in dl]
        # shapes = [list(i.shape) for i in dl]
        assert [[4, 8], [4, 1]] == shapes[0]
        assert [[4, 8], [4, 1]] == shapes[1]
        assert [[1, 8], [1, 1]] == shapes[2]
        targets = [sample[1] for sample in dl]
        assert 1 == torch.max(torch.cat(targets, dim=0))
