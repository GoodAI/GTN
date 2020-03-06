from abc import ABC, abstractmethod


class BokehComponent(ABC):
    @abstractmethod
    def create_layout(self):
        pass
