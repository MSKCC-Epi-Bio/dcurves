# dcurves

[![PyPI Latest Release](https://img.shields.io/pypi/v/dcurves.svg)](https://pypi.org/project/dcurves/)
[![PyPI Downloads Overall](https://static.pepy.tech/badge/dcurves)](https://pepy.tech/projects/dcurves)

## Table of Contents

- [dcurves](#dcurves)
  - [Table of Contents](#table-of-contents)
  - [Functions](#functions)
  - [Quick-and-dirty Library Documentation](#quick-and-dirty-library-documentation)
  - [Simple Tutorial](#simple-tutorial)
    - [Installation (bash)](#installation-bash)
    - [DCA Example](#dca-example)
  - [In-depth Tutorial and Explanations with Examples, Relevant Literature, and Forumn for Personalized Help](#in-depth-tutorial-and-explanations-with-examples-relevant-literature-and-forumn-for-personalized-help)
  - [Contributing](#contributing)
  - [Contributors](#contributors)
  - [License](#license)

Diagnostic and prognostic models are typically evaluated with measures of accuracy that do not address clinical consequences. Decision-analytic techniques allow assessment of clinical outcomes, but often require collection of additional information that may be cumbersome to apply to models that yield continuous results.

Decision Curve Analysis is a method for evaluating and comparing prediction models that incorporates clinical consequences, requiring only the data set on which the models are tested, and can be applied to models that have either continuous or dichotomous results.

## Functions

dcurves is a Python package for performing Decision Curve Analysis (DCA). It evaluates and compares prediction models for both binary and survival outcomes.

Main functions:

- `dca()`: Performs Decision Curve Analysis, calculating net benefit and interventions avoided
- `plot_graphs()`: Visualizes DCA results
- `load_test_data()`: Provides sample data for testing and examples

## Quick-and-dirty Library Documentation

https://mskcc-epi-bio.github.io/dcurves/

## Simple Tutorial

This tutorial will guide you through installing and using the `dcurves` package to perform Decision Curve Analysis (DCA) with sample cancer diagnosis data.

### Installation (bash)

```bash
# Install dcurves for DCA
pip install dcurves
```

### DCA Example

```python
# Import Libraries
from dcurves import dca, plot_graphs, load_test_data

# Load Package Simulation Data
df_binary = load_test_data.load_binary_df()

# Perform Decision Curve Analysis
df_dca = \
        dca(
            data=df_binary,
            outcome='cancer',
            modelnames=['famhistory'],
            thresholds=np.arange(0, 0.36, 0.01),
        )

# Standard DCA Plot
plot_graphs(
    plot_df=df_dca,
    graph_type='net_benefit',
    y_limits=[-0.05, 0.2]
)

```

![DCA Plot](https://github.com/MSKCC-Epi-Bio/dcurves/raw/main/public/simple_binary_dca.png)

## In-depth Tutorial and Explanations with Examples, Relevant Literature, and Forumn for Personalized Help

Visit <https://www.decisioncurveanalysis.org>

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Contributors

- Shaun Porwal (<shaun.porwal@gmail.com>)
- Rohan Singh (<singhrohan@outlook.com>)

## License

[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)
