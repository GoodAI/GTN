import tempfile
from pathlib import Path


class TempFile(tempfile.TemporaryDirectory):
    """Creates context with temp file name in a temporary directory, deletes the dir after exiting the context

    returns the file
    """

    def __init__(self, filename: str):
        super().__init__()
        self.file = Path(self.name) / filename

    def __enter__(self):
        super().__enter__()
        return self.file

    def __exit__(self, exc, value, tb):
        super().__exit__(exc, value, tb)


class TempFileName(TempFile):
    """Creates context with temp file name in a temporary directory, deletes the dir after exiting the context

    returns name of the file
    """

    def __enter__(self):
        return str(super().__enter__())