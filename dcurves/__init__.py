"""
Decision Curve Analysis in Python

Functions exported by this package:

- `dca`
- `plot_graphs`
- `load_test_data.load_binary_df`
- `load_test_data.load_survival_df`
- `load_test_data.load_case_control_df`

"""

import os
from dcurves import load_test_data
from dcurves.dca import dca
from dcurves.plot_graphs import plot_graphs

__version__ = "1.1.2.4"

data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
