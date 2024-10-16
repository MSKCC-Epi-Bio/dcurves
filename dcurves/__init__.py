"""
Decision Curve Analysis in Python

Modules exported by this package:

- `dca`
- `plot_graphs`
- `prevalence`
- `risks`

"""

__version__ = "1.0.7.0"

import os
from dcurves import load_test_data
from dcurves.dca import dca
from dcurves.plot_graphs import plot_graphs

data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
