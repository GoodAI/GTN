from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import List, Dict, Set, Optional

from matplotlib.figure import Figure
from torch import Tensor
import pandas as pd


@dataclass
class ObserverPlot:
    name: str
    figure: Figure


@dataclass
class ObserverScalar:
    name: str
    value: float


@dataclass
class ObserverTensor:
    name: str
    value: Tensor


class ObserverLevel(IntEnum):
    training = 1
    testing = 2
    inference = 3

    @classmethod
    def should_log(cls, run_level: Optional['ObserverLevel'], log_level: Optional['ObserverLevel']):
        return run_level is None or log_level is None or run_level >= log_level


class Observer:
    plots: List[ObserverPlot]
    scalars: List[ObserverScalar]
    tensors: List[ObserverTensor]

    def __init__(self, run_level: Optional[ObserverLevel] = None, run_flags: Set = None):
        self.run_level = run_level
        self.run_flags = run_flags or set()
        self.plots = []
        self.scalars = []
        self.tensors = []

    def add_plot(self, name: str, figure: Figure, log_level: Optional[ObserverLevel] = None):
        if ObserverLevel.should_log(self.run_level, log_level):
            self.plots.append(ObserverPlot(name, figure))

    def add_scalar(self, name: str, value: float, log_level: Optional[ObserverLevel] = None):
        if ObserverLevel.should_log(self.run_level, log_level):
            self.scalars.append(ObserverScalar(name, value))

    def add_tensor(self, name: str, tensor: Tensor, log_level: Optional[ObserverLevel] = None):
        if ObserverLevel.should_log(self.run_level, log_level):
            self.tensors.append(ObserverTensor(name, tensor))

    def tensors_as_dict(self) -> Dict[str, Tensor]:
        return {t.name: t.value for t in self.tensors}

    def scalars_as_dict(self) -> Dict[str, float]:
        return {t.name: t.value for t in self.scalars}

    def with_suffix(self, suffix: str) -> 'Observer':
        return SuffixObserver(self, suffix)


class ObserverWrapper(Observer):
    _observer: Observer

    def __init__(self, observer: Observer):
        super().__init__()
        self._observer = observer

    def add_plot(self, name: str, figure: Figure, log_level: Optional[ObserverLevel] = None):
        self._observer.add_plot(self._process_name(name), figure)

    def add_scalar(self, name: str, value: float, log_level: Optional[ObserverLevel] = None):
        self._observer.add_scalar(self._process_name(name), value)

    def add_tensor(self, name: str, tensor: Tensor, log_level: Optional[ObserverLevel] = None):
        self._observer.add_tensor(self._process_name(name), tensor)

    @abstractmethod
    def _process_name(self, name: str) -> str:
        pass


class SuffixObserver(ObserverWrapper):
    def __init__(self, observer: Observer, suffix: str):
        super().__init__(observer)
        self._suffix = suffix

    def _process_name(self, name: str) -> str:
        return name + self._suffix


class MultiObserver:
    observers: List[Observer]

    def __init__(self, run_level: Optional[ObserverLevel] = None, add_observer: bool = True):
        self._run_level = run_level
        self.observers = []
        if add_observer:
            self.add_observer()

    @property
    def current(self) -> Observer:
        return self.observers[-1]

    def add_observer(self) -> Observer:
        observer = Observer(self._run_level)
        self.observers.append(observer)
        return observer

    def scalars_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([o.scalars_as_dict() for o in self.observers])


class CompoundObserver:
    main: Observer
    rollout: MultiObserver

    def __init__(self, run_level: Optional[ObserverLevel] = None):
        self.main = Observer(run_level)
        self.rollout = MultiObserver(run_level, add_observer=False)
