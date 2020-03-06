from bokeh.palettes import Dark2_5 as palette
import itertools
from bokeh.plotting import Figure
from bokeh.models import HoverTool, ColumnDataSource
from pathlib import Path
from bokeh.embed import file_html

from typing import List, Optional
import pandas as pd
from bokeh.resources import CDN


class BokehPlot:

    @staticmethod
    def _sanitize_column_name(name: str) -> str:
        return name.replace(' ', '_').replace(':', '_')

    @staticmethod
    def plot_df(df: pd.DataFrame, legend: Optional[List[str]] = None, width: int = 1200, height: int = 700,
                title: str = 'loss', x_label: str = 'epoch') -> Figure:
        colors = itertools.cycle(palette)
        df = df.copy()

        df.columns = [str(i) for i in df.columns]
        if legend is None:
            legend = df.columns

        df.columns = [BokehPlot._sanitize_column_name(i) for i in df.columns]
        ds = ColumnDataSource(df)

        fig = Figure(y_axis_type="log", width=width, height=height, title=title, x_axis_label=x_label)
        fig.below[0].formatter.use_scientific = False
        for column, color, legend in zip(df.columns, colors, legend):
            glyph = fig.line(x='index', y=column, source=ds, color=color, legend_label=legend)
            # fig.add_tools(HoverTool(tooltips=[(f"{column} - {legend}", f"@{column}")], mode='vline', renderers=[glyph]))
            fig.add_tools(HoverTool(tooltips=[(f"{legend}", f"@{column}")], mode='vline', renderers=[glyph]))

        fig.legend.location = 'bottom_left'
        fig.legend.click_policy = "hide"
        fig.legend.label_text_font_size = '12px'
        fig.legend.spacing = -4
        fig.legend.background_fill_alpha = 0.6
        return fig

    @staticmethod
    def save_to_file(fig: Figure, file: Path, title: str = ""):
        with file.open('w') as f:
            f.write(file_html(fig, CDN, title))
