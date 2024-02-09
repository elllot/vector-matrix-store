from pathlib import Path

with open(Path(__file__).absolute().parents[0] / "VERSION") as _f:
    __version__ = _f.read().strip()

import logging
from logging import NullHandler

logging.getLogger(__name__).addHandler(NullHandler())
