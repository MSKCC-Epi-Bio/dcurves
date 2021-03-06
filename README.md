# dcurves

Dcurves is a Python library for Andrew Vickers' Decision Curve Analysis method to evaluate prediction models and diagnostic tests. 


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dcurves.

```bash
pip install dcurves
```
## Usage

```python
import dcurves

# load provided simulation dataset(s)

# load (1) binary endpoint data
df_binary = dcurves.load_test_data.load_binary_df()
# load (2) survival endpoint data
df_surv = dcurves.load_test_data.load_survival_df()
# load (3) case-control endpoint data
df_case_control = dcurves.load_test_data.load_case_control_df()
# load (4) 2nd binary endpoint data 
df_cancer_dx = dcurves.load_test_data.load_cancerdx_df()

# run decision curve analysis on dataset of choice

# (1) binary endpoint data
binary_output_df = dcurves.dca(
        data = df_binary,
        outcome = 'cancer',
        predictors = ['cancerpredmarker', 'marker'],
        thresh_vals = [0.01, 1.0, 0.01],
        probabilities = [False, True]
)

# (2) survival endpoint data

survival_output_df = dcurves.dca(
    data = df_surv,
    outcome = 'cancer',
    predictors = ['cancerpredmarker'],
    thresh_vals = [0.01, 1.0, 0.01],
    probabilities = [False],
    time = 1,
    time_to_outcome_col = 'ttcancer'
)

# (4) 2nd binary endpoint data

dan_test_output_df = dcurves.dca(
    data = df_cancer_dx,
    outcome = 'cancer',
    predictors = ['famhistory'],
    thresh_vals = [0.01, 1.0, 0.01],
    probabilities = [False]
)

# plot DCA results for binary endpoint (ideally in a jupyter/other .ipynb notebook

dcurves.plot_net_benefit_graphs(binary_output_df, y_limits=[-0.05, 0.2], color_names=['lightgreen', 'blue', 'red'])

```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache 2.0]([https://choosealicense.com/licenses/apache-2.0/])
