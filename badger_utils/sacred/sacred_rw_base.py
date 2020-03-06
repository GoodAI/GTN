import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

import torch


class SacredRWBase(ABC):
    """Common parent to the writer and reader"""

    _experiment_dirs_created: bool = False
    MODEL_SUFFIX = '.model'

    @property
    @abstractmethod
    def experiment_id(self) -> int:
        pass

    @property
    def experiment_data_dir(self) -> Path:
        # noinspection PyProtectedMember
        results_dir = Path.cwd() / 'data' / 'results' / str(self.experiment_id)
        if not self._experiment_dirs_created:
            if not results_dir.exists():
                results_dir.mkdir(parents=True)
            self._experiment_dirs_created = True
        return results_dir

    @staticmethod
    def create_artifact_name(name: str, epoch: int) -> str:
        parts = name.split('.')
        extension = '.'.join([''] + parts[1:])
        return f'{parts[0]}_ep_{epoch}{extension}'

    @staticmethod
    def _dict_to_blob(dictionary: Dict[str, Any]) -> bytes:
        """ Serialize the dictionary to blob in two levels.

        Returns: serialized object storing all the data.
        """

        def to_bytes(data: Any) -> bytes:
            writer = io.BytesIO()
            torch.save(data, writer)
            return writer.getvalue()

        # convert each value of the dictionary to bytes
        target_dict = {key: to_bytes(value) for (key, value) in dictionary.items()}

        # serialize the target_dict as well, return the data
        return to_bytes(target_dict)

    @staticmethod
    def _blob_to_dict(blob: bytes) -> Dict[str, Any]:
        """ Deserialize the dictionary from the blob.

        Returns: dictionary containing objects to be loaded by the SerializableModel
        """

        def from_bytes(my_blob: bytes) -> Dict[str, bytes]:
            data_buff = io.BytesIO(my_blob)
            dictionary = torch.load(data_buff)
            return dictionary

        top_dict = from_bytes(blob)
        result = {key: from_bytes(value) for (key, value) in top_dict.items()}

        return result
