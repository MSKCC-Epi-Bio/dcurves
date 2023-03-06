## Installation

- Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dcurves
- While this is the quick-and-dirty method to install a package such as `dcurves` into your local environment, you should use a virtual environment and make sure your dependencies are compatible while using `dcurves`

```bash
pip install dcurves
```

## Usage
### Import dcurves and numpy
```python
from dcurves import dca, plot_graphs, load_test_data
import numpy as np
```
### Usage - Binary Outcomes
```python
from dcurves import dca, plot_graphs, load_test_data
import numpy as np

dca_results = \
    dca(
        data=load_test_data.load_binary_df(),
        outcome='cancer',
        modelnames=['famhistory'],
        thresholds=np.arange(0,0.46,0.01)
    )

plot_graphs(
    plot_df=dca_results,
    graph_type='net_benefit',
    y_limits=[-0.05, 0.15],
    color_names=['blue', 'red', 'green']
)
```
### Usage - Survival Outcomes
```python
from dcurves import dca, plot_graphs, load_test_data
import numpy as np

dca_results = \
    dca(
        data=load_test_data.load_survival_df(),
        outcome='cancer',
        modelnames=['famhistory', 'marker', 'cancerpredmarker'],
        models_to_prob=['marker'],
        thresholds=np.arange(0,0.46,0.01),
        time_to_outcome_col='ttcancer',
        time=1
    )

plot_graphs(
    plot_df=dca_results,
    graph_type='net_benefit',
    y_limits=[-0.025, 0.175],
    color_names=['blue', 'red', 'green', 'purple', 'black']
)
```