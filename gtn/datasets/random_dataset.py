import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    def __init__(self, item_size: int, count: int, device: str):
        self.device = device
        self.item_size = item_size
        self.count = count

    def __getitem__(self, item: int):
        if item >= self.count:
            raise IndexError(f'Index {item} is out of range. Total items: {self.count}')
        return torch.rand((self.item_size,), device=self.device)

    def __len__(self) -> int:
        return self.count


class RandomDatasetWithTargets(Dataset):
    def __init__(self, random_item_size: int, count: int, target_classes: int, device: str):
        self.device = device
        self.target_classes = target_classes
        self.random_item_size = random_item_size
        self.count = count

    def __getitem__(self, item: int):
        if item >= self.count:
            raise IndexError(f'Index {item} is out of range. Total items: {self.count}')
        return torch.rand((self.random_item_size,), device=self.device), torch.randint(self.target_classes, (1,), device=self.device)

    def __len__(self) -> int:
        return self.count
