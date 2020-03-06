import pytest
from badger_utils.torch.test_utils import same

from gtn.datasets.datasets import Datasets
from gtn.datasets.random_dataset import RandomDataset


class TestRandomDataset:
    def test_dataset(self):
        ds = RandomDataset(16, 3, 'cpu')
        assert 3 == len(ds)
        assert [16] == list(ds[0].shape)
        assert [16] == list(ds[1].shape)
        assert [16] == list(ds[2].shape)
        assert not same(ds[0], ds[1])
        assert not same(ds[0], ds[2])
        assert not same(ds[1], ds[2])

    def test_dataset_out_of_range(self):
        ds = RandomDataset(16, 3, 'cpu')
        with pytest.raises(IndexError):
            _ = ds[3]

    def test_dataloader(self):
        dl = Datasets.random_dataloader(4, 8, 9, 'cpu')
        shapes = [list(i.shape) for i in dl]
        assert [4, 8] == shapes[0]
        assert [4, 8] == shapes[1]
        assert [1, 8] == shapes[2]
