from abc import abstractmethod
from typing import Optional

from badger_utils.torch.device_aware import DeviceAwareModule


class Learner(DeviceAwareModule):
    def __init__(self, device: Optional[str] = None):
        super().__init__(device)

    @abstractmethod
    def reset(self):
        pass
