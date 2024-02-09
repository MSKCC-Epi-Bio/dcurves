"""
Decision Curve Analysis in Python

Functions exported by this package:

- `dca`
- `plot_graphs`
- `load_binary_df`
- `load_survival_df`
- `load_case_control_df`

"""

__version__ = "1.0.6.4"

import os
from dcurves import load_test_data
from dcurves.dca import dca
from dcurves.plot_graphs import plot_graphs

data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
