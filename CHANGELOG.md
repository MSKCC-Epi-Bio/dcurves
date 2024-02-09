# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic 
Versioning](https://semver.org/spec/v2.0.0.html).

Starting from version 1.0.6.5, this project uses [*towncrier*](https://towncrier.readthedocs.io/) for managing changelog entries.

## [1.0.6.5] - Release Date

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

## Previous Versions

Changes for versions prior to 1.0.6.5 were not managed with towncrier. Please refer to the commit history or previous release notes for details on 
those versions.

