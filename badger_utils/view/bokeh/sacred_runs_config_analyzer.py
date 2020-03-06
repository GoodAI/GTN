from typing import List, Callable, Optional

from badger_utils.view.bokeh.bokeh_component import BokehComponent
from bokeh.layouts import column, row
from bokeh.models import TextInput, Div, DataTable, ColumnDataSource, TableColumn

from badger_utils.sacred import SacredConfig, SacredUtils


class SacredRunsConfigAnalyzer(BokehComponent):
    _sacred_utils: SacredUtils
    _on_run_selected: Optional[Callable[[int], None]]

    def __init__(self, sacred_config: SacredConfig, on_run_selected: Optional[Callable[[int], None]],
                 min_id: int = 2258, max_id: int = 2325):
        self._sacred_config = sacred_config
        self._on_run_selected = on_run_selected
        self._sacred_utils = SacredUtils(self._sacred_config)
        self._min_id = min_id
        self._max_id = max_id

    def _update_by_input(self):
        try:
            min_id = int(self.widget_min_id.value)
            max_id = int(self.widget_max_id.value)
            diff_result = self._sacred_utils.config_diff(list(range(min_id, max_id + 1)))
            formatted_config = diff_result.common_as_text('<br/>')
            df = diff_result.diff_as_df()
            self.ds_common.data = df
            self.widget_dt_common.columns = [TableColumn(field=c, title=c) for c in df.columns]
            self.widget_config_common.text = f'<pre>{formatted_config} <hr/></pre>'
            # self.widget_config_common.text = f'<pre>{formatted_config} <hr/>{formatted_diff}</pre>'
        except Exception as e:
            print(f'Exception: {e}')

    def _on_table_click(self, p):
        run_ids = self.ds_common.data['run_ids'][p][0]
        if self._on_run_selected is not None:
            self._on_run_selected(run_ids[0])

    def create_layout(self):
        self.ds_common = ColumnDataSource({'a': [4]})
        self.ds_common.selected.on_change('indices', lambda a, o, n: self._on_table_click(n))
        self.widget_min_id = TextInput(title='Min Id', value=str(self._min_id))
        self.widget_min_id.on_change('value', lambda a, o, n: self._update_by_input())
        self.widget_max_id = TextInput(title='Max Id', value=str(self._max_id))
        self.widget_max_id.on_change('value', lambda a, o, n: self._update_by_input())
        self.widget_config_common = Div(text='')
        self.widget_dt_common = DataTable(source=self.ds_common, width=1000)
        self._update_by_input()
        return column(
            row(self.widget_min_id, self.widget_max_id),
            self.widget_config_common,
            self.widget_dt_common
        )
