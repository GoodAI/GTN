import io
import base64
from typing import Dict

import PIL

from PIL.Image import Image
from matplotlib.figure import Figure


class ImageUtils:
    """
        Image conversion utilities

        Examples:
            IPython HTML display of pd.DataFrame:
                df['loss_plot'] = df['run_id'].map(lambda r: ImageUtils.figure_to_image(plot_loss(r)))
                display.display(HTML(df.to_html(formatters={'loss_plot': ImageUtils.image_to_html}, escape=False)))
            Bokeh DataTable display:
                df['loss_plot'] = df['run_id'].map(lambda r: ImageUtils.figure_to_html(plot_loss(r)))
                show(DataTableBuilder(df).create())
    """

    @staticmethod
    def image_base64(im: Image) -> str:
        """
        Convert image to base64 encoded JPEG
        Args:
            im: Image to be converted

        Returns:
            Base64 string of JPEG image
        """
        with io.BytesIO() as buffer:
            im.save(buffer, 'jpeg')
            return base64.b64encode(buffer.getvalue()).decode()

    @staticmethod
    def image_to_html(im: Image, attributes: Dict[str, str] = None) -> str:
        """
        Convert Image to HTML inline IMG tag
        Args:
            im: Image to be converted
            attributes: Dict with HTML attributes to be inserted to <img> tag

        Returns:
            String with inline IMG tag with base64 encoded JPEG image
        """
        a = '' if attributes is None else ' '.join([f'{k}="{v}"' for k, v in attributes.items()])
        return f'<img {a} src="data:image/jpeg;base64,{ImageUtils.image_base64(im)}">'

    @staticmethod
    def figure_to_image(fig: Figure) -> Image:
        """
        Convert matplot Figure to PIL Image
        Args:
            fig: Figure to be converted

        Returns:
            PIL Image
        """
        canvas = fig.canvas
        canvas.draw()
        return PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())

    @staticmethod
    def figure_to_html(fig: Figure, attributes: Dict[str, str] = None) -> str:
        """
        Convert matplot Figure to HTML inline IMG tag
        Args:
            fig: Figure to be converted
            attributes: Dict with HTML attributes to be inserted to <img> tag

        Returns:
            String with inline IMG tag with base64 encoded JPEG image
        """
        return ImageUtils.image_to_html(ImageUtils.figure_to_image(fig), attributes)
