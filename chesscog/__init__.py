"""Chess position inference using computer vision."""

import logging
import sys

from .__version__ import __version__
from .core import io as _
from .recognition import ChessRecognizer

__all__ = ["ChessRecognizer"]


def _setup_logger(level: int = logging.INFO):
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


_setup_logger()
