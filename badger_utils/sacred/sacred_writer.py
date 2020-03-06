from pathlib import Path

import torch
from badger_utils.view import MultiObserver
from badger_utils.view.observer_utils import Observer
from matplotlib.figure import Figure
from torch import Tensor
from PIL import Image
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from badger_utils.file.temp_file import TempFile, TempFileName
from badger_utils.sacred.sacred_config import SacredConfig
from badger_utils.sacred.sacred_rw_base import SacredRWBase
from badger_utils.sacred.serializable_model import Serializable


class SacredWriter(SacredRWBase):
    """Writes data to the Sacred"""

    _experiment: Experiment

    def __init__(self, experiment: Experiment, sacred_config: SacredConfig):
        super().__init__()
        self._experiment = experiment
        self._experiment_setup()
        self._init_observers(sacred_config)

    @property
    def experiment_id(self) -> int:
        # noinspection PyProtectedMember
        return self._experiment.current_run._id

    def save_tensor(self, data: Tensor, name: str, epoch: int):
        with TempFile(self.create_artifact_name(f'{name}.pt', epoch)) as file:
            with file.open('wb') as f:
                torch.save(data, f)
            self._add_binary_file(file)

    def save_model(self, model: Serializable, name: str, epoch: int):
        dictionary = model.serialize()
        blob = self._dict_to_blob(dictionary)

        # create temp directory, write blob to the file, push file to the sacred, delete directory
        with TempFile(self.create_artifact_name(name + SacredRWBase.MODEL_SUFFIX, epoch)) as file:
            with file.open('wb') as f:
                f.write(blob)
            self._add_binary_file(file)

    def save_scalar(self, value: float, name: str, epoch: int):
        self._experiment.log_scalar(name, value, epoch)

    # noinspection PyShadowingBuiltins
    def save_matplot_figure(self, figure: Figure, name: str, epoch: int, format: str = 'svg'):
        with TempFileName(self.create_artifact_name(name, epoch)) as filename:
            figure.savefig(fname=filename, format=format)
            self._experiment.add_artifact(filename)

    def save_image(self, image: Image, name: str, epoch: int):
        with TempFile(self.create_artifact_name(name, epoch)) as file:
            with file.open('wb') as f:
                image.save(f)
            self._experiment.add_artifact(str(file))

    def _experiment_setup(self):
        self._experiment.captured_out_filter = lambda captured_output: "Output capturing turned off."

    def _init_observers(self, database_config: SacredConfig):
        observer = database_config.create_mongo_observer()
        self._experiment.observers.append(observer)

    def _add_binary_file(self, file: Path):
        self._experiment.add_artifact(filename=str(file), content_type='application/octet-stream')

    def save_observer(self, observer: Observer, epoch: int):
        for s in observer.scalars:
            self.save_scalar(s.value, s.name, epoch)
        for t in observer.tensors:
            self.save_tensor(t.value, t.name, epoch)
        for p in observer.plots:
            self.save_matplot_figure(p.figure, p.name, epoch)
