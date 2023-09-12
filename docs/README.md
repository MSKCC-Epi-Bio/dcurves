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

## Table Of Contents

The documentation follows the best practice for
project documentation as described by Daniele Procida
in the [Diátaxis documentation framework](https://diataxis.fr/)
and consists of four separate parts:

1. [Tutorials](tutorials.md)
2. [How-To Guides](how-to-guides.md)
3. [Reference](reference.md)
4. [Explanation](explanation.md)

Quickly find what you're looking for depending on
your use case by looking at the different pages.

## Project Overview

::: dcurves

## In-depth tutorial and explanations:

###### https://www.decisioncurveanalysis.org 


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
###### building newsfragments example (dryrun with draft): poetry run towncrier build --draft --version 1.0.6.4
###### Creating custom towncrier types: check pyproject.toml for example, as all of those are custom types
