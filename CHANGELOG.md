# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This project uses [*towncrier*](https://towncrier.readthedocs.io/) and the changes for the upcoming release can be found in <https://github.com/twisted/my-project/tree/main/changelog.d/\>.

<!-- towncrier release notes start -->


### Improvement

- Achieved 100% test coverage. GH-18
- In GitHub Actions, adjusted the sequence to run pylint before pytest. Additionally, pylint now runs exclusively on the 'dcurves' directory, excluding tests. GH-19


### Feature

- Updated to support the latest Pandas version. GH-18


### Addition

- Added support for Python 3.11. GH-18


### Removal

- Removed support for Python 3.8. GH-18
