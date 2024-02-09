## Contributors


- Shaun Porwal (shaun.porwal@gmail.com)
- Rohan Singh (singhrohan@outlook.com)


# dcurves
Diagnostic and prognostic models are typically evaluated with measures of accuracy that do not address clinical
consequences. Decision-analytic techniques allow assessment of clinical outcomes, but often require collection of
additional information that may be cumbersome to apply to models that yield continuous results. Decision Curve
Analysis is a method for evaluating and comparing prediction models that incorporates clinical consequences,
requiring only the data set on which the models are tested, and can be applied to models that have either continuous or
dichotomous results. The dca function performs decision curve analysis for binary and survival outcomes. Review the
DCA tutorial (towards the bottom) for a detailed walk-through of various applications. Also, see
www.decisioncurveanalysis.org for more information.


## Project Overview


::: dcurves


## In-depth tutorial, explanations, peer-reviewed literature, and discussion board:


###### https://www.decisioncurveanalysis.org 


## Changelog

## Current Version: [1.1.0] - 02/09/24

### Feature
- Merge split_dca with main branch. GH-6
- Incorporated Lowess smoothing and added `smooth_frac` parameter to control smoothing degree in plot functions. GH-23

### Improvement
- Use new site for tutorial, not Dan's site. GH-14
- Update README.md. GH-15
- GitHub actions now lint only the 'dcurves/' directory, excluding tests, and run pylint before pytest. GH-20
- Data Loading, Cross Validation updated, and added `show_grid` option. GH-21
- Added options to display legend and grid in plots. GH-23
- Added plot quality and saving settings for enhanced user control over plot outputs. GH-23
- Updated docstrings across various modules for better clarity and documentation. GH-23
- Enhanced data validation with new checks and corresponding unit tests for improved reliability. GH-23

### Bugfix
- Actions tox fix. GH-13
- Fixed GitHub actions and added release notes system using towncrier. GH-20

### Addition
- Update to include latest Pandas version, bring tests to 100% coverage. GH-18
- Added support for Python 3.11. GH-18

### Removal
- Removed support for Python 3.8. GH-18

For a detailed list of changes in each release, please refer to the [CHANGELOG.md]( https://github.com/MSKCC-Epi-Bio/dcurves/blob/main/CHANGELOG.md).


## Contributing


###### Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change


###### Please make sure to update tests as appropriate


## License


[Apache 2.0]([https://choosealicense.com/licenses/apache-2.0/])

## Note


###### setup.py is deprecated now that dependencies are managed by `poetry` package manager


## Using towncrier to document changes for this project


###### creating a newsfragment example (format is PR/Issue#.type for newsfragment): poetry run towncrier create -c "Added support for Python 3.11." 18.addition
###### Make sure the newsfragments are tracked by git before building
###### building newsfragments example (dryrun with draft): poetry run towncrier build --draft --version 1.1.0
###### Creating custom towncrier types: check pyproject.toml for example, as all of those are custom types
