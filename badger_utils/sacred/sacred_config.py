import ssl
from abc import abstractmethod
from typing import Callable

from sacred.observers import MongoObserver


class SacredConfig:

    @abstractmethod
    def create_mongo_observer(self) -> MongoObserver:
        pass


class SacredConfigImpl(SacredConfig):
    def __init__(self, observer_factory: Callable[[], MongoObserver]):
        self._observer_factory = observer_factory

    def create_mongo_observer(self) -> MongoObserver:
        return self._observer_factory()


class SacredConfigFactory:
    @classmethod
    def shared(cls, legacy: bool = False) -> SacredConfig:
        # Add your real IP address
        mongo_url = f'mongodb://192.168.1.1:27017/'

        def observer():
            return MongoObserver(url=mongo_url, db_name='sacred', ssl=True)

        return SacredConfigImpl(observer)

    @classmethod
    def local(cls, docker: bool = True) -> SacredConfig:
        def observer():
            return MongoObserver(url=f'mongodb://sample:password@localhost:27017/?authMechanism=SCRAM-SHA-1',
                                 db_name='db')

        return SacredConfigImpl(observer if docker else lambda: MongoObserver())
