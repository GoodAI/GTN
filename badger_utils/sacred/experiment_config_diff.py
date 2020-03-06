from typing import Dict, Any, List, Tuple

import pandas as pd

from badger_utils.view.config_utils import tuple_to_dict


class ExperimentConfigDiff:
    _diff: Dict[List[Tuple[str, Any]], List[int]]
    _common: Dict[str, Any]

    def __init__(self, common: Dict[str, Any], diff: Dict[List[Tuple[str, Any]], List[int]]):
        """

        Args:
            common: dict of config vars, e.g. {'size': 10, 'epochs': 1000}
            diff: dict with keys being list of tuples of ('name', 'value') of config and list of run_ids as value,
                  e.g. {[('n_experts', 4), ('n_inputs', 3)]: [23, 24], [('n_experts', 4), ('n_inputs', 2)]: [25]}
        """
        self._common = common
        self._diff = diff

    def diff_as_df(self, explode_by_run_id: bool = False) -> pd.DataFrame:
        """
        Returns:
            DataFrame with columns named by config keys
            plus one column "run_ids" where are stored comma separated run_ids
        """
        df = pd.DataFrame([{**tuple_to_dict(r), **{'run_ids': v}} for r, v in self._diff.items()])

        if explode_by_run_id:
            df = df.explode('run_ids').astype({'run_ids': int}).set_index('run_ids')
            df.index.name = None

        return df

    def diff_as_lines(self) -> List[str]:
        """
        Returns:
            List of one_line string representation for diff. Usable e.g. for a plot legend.

        """

        return ExperimentConfigDiff.df_as_lines(self.diff_as_df())

    def common_as_text(self, line_delimiter: str = '\n') -> str:
        return line_delimiter.join([f'{k}: {v}' for k, v in self._common.items()])

    def diff_filtered_run_ids(self, filter_dict: Dict[str, Any]) -> List[int]:
        """
        Return list of run_ids for runs that match filter_dict. Only runs matching all filter conditions are selected.
        Args:
            filter_dict: Dict config_item -> expected_value. E.g. {'n_experts': 4, 'rollout_size': 8}

        Returns:
            List of run_ids
        """
        filtered = self.filter_df(self.diff_as_df(), filter_dict)
        return self.flatten(filtered['run_ids'])

    @staticmethod
    def filter_df(df: pd.DataFrame, filter_dict: Dict[str, Any]) -> pd.DataFrame:
        for k, v in filter_dict.items():
            df = df.loc[df[k] == v]
        return df

    @staticmethod
    def flatten(l):
        return [item for sublist in l for item in sublist]

    @staticmethod
    def df_as_lines(df: pd.DataFrame) -> List[str]:
        """
        Convert DataFrame to list of strings representation
        Args:
            df: DataFrame to be converted

        Returns:
            List of one_line string representation for DataFrame. Usable e.g. for a plot legend.

        """

        def format_config(r):
            return ', '.join([f'{c}: {v}' for c, v in zip(r._fields, r)])

        return [format_config(r) for r in df.itertuples(index=False, name='Row')]

    @staticmethod
    def df_as_description_runids_dict(df: pd.DataFrame) -> Dict[str, List[int]]:
        result = {}
        for idx, row in df.iterrows():
            columns_values = [f'{name}: {row[name]}' for name in row.index if name != 'run_ids']
            description = ', '.join(columns_values)
            result[description] = row['run_ids']
        return result
