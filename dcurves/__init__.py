"""
Decision Curve Analysis in Python

Functions exported by this package:

- `dca` - Decision curve analysis for binary and survival outcomes
- `plot_graphs` - Plot net benefit or net intervention avoided

"""

import os
from dcurves.dca import dca
from dcurves.plot_graphs import plot_graphs

# ---------------------------------------------------------------------------
# Dynamic version: sourced from installed package metadata to avoid
# hard-coding and drift. Falls back to a placeholder when running directly
# from a checkout without installation.
# ---------------------------------------------------------------------------

from importlib import metadata as _metadata

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # package not installed (e.g. source tree)
    __version__ = "0.0.0.dev0"

data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
