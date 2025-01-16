# eerfile

[![License](https://img.shields.io/pypi/l/eerfile.svg?color=green)](https://github.com/alisterburt/eerfile/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/eerfile.svg?color=green)](https://pypi.org/project/eerfile)
[![Python Version](https://img.shields.io/pypi/pyversions/eerfile.svg?color=green)](https://python.org)
[![CI](https://github.com/alisterburt/eerfile/actions/workflows/ci.yml/badge.svg)](https://github.com/alisterburt/eerfile/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alisterburt/eerfile/branch/main/graph/badge.svg)](https://codecov.io/gh/alisterburt/eerfile)

Read electron event representation (EER) files as NumPy arrays in Python.

## Installation

*eerfile* is available on PyPI and can be installed with `pip`

```shell
pip install eerfile
```

## Usage

```python
import eerfile

# render multi-frame micrograph from file and target dose per frame
# dose per frame is in electrons per square angstrom
image = eerfile.render("FoilHole_19622436.eer", dose_per_output_frame=1.0)

# or you can read the entire stack of EER frames
eer_frames = eerfile.read("FoilHole_19622436.eer")
```

## Acknowledgements

This package is a very thin convenience layer on top of [`tifffile`](https://github.com/cgohlke/tifffile/)
which provides the EER frame decoding logic.
