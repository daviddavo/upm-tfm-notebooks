"""
Utils for use in Jupyter Notebooks
"""

import io
from matplotlib import pyplot as plt

def download_matplotlib(fig: plt.Figure, filename):
    import solara
    format = filename.split('.')[-1]

    def get_data():
        buf = io.BytesIO()
        fig.savefig(buf, format=format, bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        return buf

    return solara.FileDownload(
        data=get_data,
        filename=filename,
    )

def download_plotly(fig, filename):
    import solara
    format = filename.split('.')[-1]

    return solara.FileDownload(
        data=lambda: fig.to_image(format=format),
        filename=filename,
    )

