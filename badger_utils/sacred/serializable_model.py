from abc import abstractmethod
from typing import Dict, Any


class Serializable:
    """Interface for an arbitrary class being able to serialize/deserialize to/from python dictionary
    using objects that can be pickled/depickled"""

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """
        Returns: dictionary of objects to be serialized
        """
        raise NotImplementedError()

    @abstractmethod
    def deserialize(self, data: Dict[str, Any]):
        """ Given the dictionary, write 'object' data to the appropriate places
        """
        raise NotImplementedError()
