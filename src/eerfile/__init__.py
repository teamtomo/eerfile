"""Read electron event representation (EER) files in Python"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eerfile")
except PackageNotFoundError:
    __version__ = "uninstalled"
__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .read_eer import read
from .render_eer import render

__all__ = [
    "read",
    "render"
]
