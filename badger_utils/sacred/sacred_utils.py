from dataclasses import dataclass
from pathlib import Path
from typing import List, Any, Dict, Optional

import pymongo

from badger_utils.sacred import SacredReader, SacredWriter
from badger_utils.sacred.experiment_config_diff import ExperimentConfigDiff
from badger_utils.sacred.sacred_config import SacredConfig
from sacred import Experiment
from sacred.observers import MongoObserver
import pandas as pd

from badger_utils.view.config_utils import find_common_keys, group_dicts, dict_omit_keys


@dataclass
class SacredRun:
    id: int
    config: Dict[str, Any]


class SacredUtils:
    _observer: MongoObserver

    def __init__(self, config: SacredConfig):
        self._observer = config.create_mongo_observer()
        self._config = config

    def load_metrics(self, run_ids: List, name: str, average_window: Optional[int] = None) -> pd.DataFrame:
        runs = self._observer.metrics.find({'run_id': {'$in': run_ids}, 'name': name})
        df = pd.DataFrame()
        for run in runs:
            steps, values = run['steps'], run['values']
            df = df.join(pd.DataFrame(values, index=steps, columns=[run['run_id']]), how='outer')

        if average_window is not None:
            return df.rolling(window=average_window).mean()
        else:
            return df

    def list_metrics(self, run_ids: List[int]) -> List[str]:
        return self._observer.metrics.distinct('name', {'run_id': {'$in': run_ids}})

    def get_last_run(self) -> SacredRun:
        runs = self._observer.runs.find({}, {'_id': 1, 'config': 1}).sort([('_id', pymongo.DESCENDING)]).limit(1)
        last_run = next(runs)
        return SacredRun(last_run['_id'], last_run['config'])

    def config_diff(self, run_ids: List[int]) -> ExperimentConfigDiff:
        """
        Analyze config dict of multiple runs and return common keys and variable config.
        Args:
            run_ids: List of run_ids

        Returns:
            (common, diff) where:
                * common is dict of config vars, e.g. {'size': 10, 'epochs': 1000}
                * diff is dict with keys being list of tuples of ('name', 'value') of config and list of run_ids as value,
                  e.g. {[('n_experts', 4), ('n_inputs', 3)]: [23, 24], [('n_experts', 4), ('n_inputs', 2)]: [25]}

        """
        runs = self._observer.runs.find({'_id': {'$in': run_ids}}, {'_id': 1, 'config': 1})
        items = list(runs)
        common_keys = find_common_keys(items, lambda x: x['config'])
        diff = group_dicts(items, lambda x: dict_omit_keys(x['config'], set(common_keys) | {'seed'}),
                           lambda x: x['_id'])
        return ExperimentConfigDiff(common_keys, diff)

    def get_reader(self, experiment_id: int, data_dir: Optional[Path] = None) -> SacredReader:
        """
        Args:
            experiment_id: id of the experiment to be loaded
            data_dir: optional directory for caching the data from sacred (will append data/loaded_from_sacred)
        """
        return SacredReader(experiment_id, self._config, data_dir)

    def get_writer(self, experiment: Experiment) -> SacredWriter:
        """
        Args:
            experiment: experiment instance
        """
        return SacredWriter(experiment, self._config)
