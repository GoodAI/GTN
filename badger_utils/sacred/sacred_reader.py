import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Pattern

import torch
from torch import Tensor
from PIL import Image

from badger_utils.sacred.sacred_config import SacredConfig
from badger_utils.sacred.sacred_rw_base import SacredRWBase
from badger_utils.sacred.serializable_model import Serializable
from badger_utils.sacred.gridfs_reader import GridFSReader


class SacredReader(SacredRWBase):
    """Reads data from the Sacred"""
    _experiment_id: int
    _grid_reader: GridFSReader

    _run_data: Optional[Dict] = None
    _local_dir: Path  # parent directory for downloaded data
    _episode_regex: Pattern[str] = re.compile(f'.*ep_([0-9]+).*$')

    def __init__(self,
                 experiment_id: int,
                 sacred_config: SacredConfig,
                 data_dir: Optional[Path] = None):
        """
        Args:
            experiment_id: id of the experiment to be loaded
            sacred_config: config of the sacred database
            data_dir: optional directory for caching the data from sacred (will append data/loaded_from_sacred)
        """
        self._experiment_id = experiment_id
        self._mongo_observer = sacred_config.create_mongo_observer()
        self._grid_reader = GridFSReader(self._mongo_observer.fs)

        self._local_dir = Path.cwd() if data_dir is None else data_dir
        assert self._local_dir.exists(), f'given data_dir does not exist: {str(self._local_dir)}'

        self._local_dir = self._local_dir / 'data' / 'loaded_from_sacred'
        if not self._local_dir.exists():
            self._local_dir.mkdir(parents=True)

    @property
    def experiment_id(self) -> int:
        return self._experiment_id

    @property
    def experiment_data_dir(self) -> Path:
        result = self._local_dir / f'artifacts-{self.experiment_id}'
        if not result.exists():
            result.mkdir(parents=True)
        return result

    @property
    def config(self) -> Dict[str, Any]:
        """Try reading from the variable, try loading locally, download and read"""
        return self.run_data['config']

    @property
    def run_data(self) -> Dict[str, Any]:
        if self._run_data is None:
            self._run_data = self._mongo_observer.runs.find_one({'_id': self.experiment_id})

        return self._run_data

    def load_tensor(self, name: str, epoch: int) -> Tensor:
        return torch.load(self._checked_artifact_file(name, epoch))

    def load_image(self, name: str, epoch: int) -> Image:
        return Image.open(self._checked_artifact_file(name, epoch))

    def load_model(self, model: Serializable, name: str, epoch: int):
        """ Try loading the data from given name_ep_{epoch}.model file to the model. Download in file not found.
        Args:
            model: model to load the data into
            name: name of the file used in the save_model method
            epoch: epoch from which to load the model
        """
        with self._checked_artifact_file(f'{name}.model', epoch).open('rb') as f:
            blob = f.read()

        dictionary = self._blob_to_dict(blob)
        model.deserialize(dictionary)

    def find_last_epoch(self) -> int:
        """Detects the last episode in (a potentially currently running) experiment and returns it"""
        return self._find_last_epoch(self._grid_reader.list_artifacts(self._experiment_id))

    def get_epochs(self) -> List[int]:
        return self._get_epochs(self._grid_reader.list_artifacts(self._experiment_id))

    @property
    def _config_file(self) -> Path:
        return self.experiment_data_dir / 'config.json'

    def _checked_artifact_file(self, name: str, epoch: int) -> Path:
        """Download artifacts when file is not found"""
        file = self.experiment_data_dir / self.create_artifact_name(name, epoch)
        if not file.exists():
            print(f'File "{file}" not found locally..')
            self._download_artifact(self.create_artifact_name(name, epoch), file.resolve())
        if not file.exists():
            raise FileNotFoundError(f"No such file: '{file}'")
        return file

    def _download_artifact(self, name: str, target_file: str):
        """Download a single artifact file using GridFS and store it locally"""

        print(f'Downloading artifact "{name}" using GridFS...')
        with open(target_file, 'wb') as w:
            w.write(self._grid_reader.read_artifact(self.experiment_id, name))

    @staticmethod
    def _find_last_epoch(files: List[str]) -> Optional[int]:
        epochs = SacredReader._get_epochs(files)
        return None if len(epochs) == 0 else epochs[-1]

    @staticmethod
    def _get_epochs(files: List[str]) -> List[int]:
        # go through the filenames, try to match the regex for a given run
        epochs = {SacredReader.parse_epoch_from_filename(f) for f in files if f.endswith(SacredRWBase.MODEL_SUFFIX)}
        return sorted(set(filter(None, epochs)))

    @staticmethod
    def parse_epoch_from_filename(filename: str) -> Optional[int]:
        # pattern to be detected: example of the name: run_12_Actor_0_ep_14000.model
        episode_match = re.match(SacredReader._episode_regex, filename)
        if episode_match is not None:
            return int(episode_match.group(1))
        else:
            return None

    def list_artifacts(self) -> List[str]:
        return self._grid_reader.list_artifacts(self.experiment_id)
