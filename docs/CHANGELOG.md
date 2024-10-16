# Changelog

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-02-09

### Added
- Support for Python 3.11.
- Lowess smoothing functionality with `smooth_frac` parameter for plot functions.
- Options to display legend and grid in plots.
- Plot quality and saving settings for enhanced user control over plot outputs.

### Changed
- Updated to support the latest Pandas version.
- Enhanced data validation with new checks and corresponding unit tests.
- Updated docstrings across various modules for better clarity.
- Improved GitHub Actions workflow: lint only 'dcurves/' directory, run pylint before pytest.

### Removed
- Support for Python 3.8.

## [1.0.6.5] - 2023-09-12

### Added
- Merged split_dca with main branch. (GH-6)
- New site for tutorial, replacing Dan's site. (GH-14)
- Show_grid option for plots. (GH-21)

### Changed
- Updated README.md. (GH-15)
- Improved data loading and cross-validation processes. (GH-21)
- Achieved 100% test coverage. (GH-18)

### Fixed
- Actions tox issues. (GH-13)
- GitHub actions and release notes system. (GH-20)

## [1.0.6.4] - 2023-09-12

- Minor patches and documentation updates.

## [1.0.6.2] - 2023-04-19
## [1.0.6.1] - 2023-04-19
## [1.0.6] - 2023-04-19
## [1.0.5] - 2023-02-07
## [1.0.4] - 2023-01-22
## [1.0.3] - 2023-01-17
## [1.0.2] - 2023-01-17
## [1.0.1] - 2023-01-17

- Various improvements and bug fixes. (Specific details not available)

## [1.0.0] - 2023-01-17

### Added
- Initial release of the Decision Curve Analysis plotting module.
- Core functionality for creating net benefit and net intervention avoided plots.

## [0.0.3] - 2022-06-01
## [0.0.2] - 2022-05-01

- Pre-release versions. Development and testing.

# SP 241016: These below are from before, above is from chatGPT, need to modify still and then can delete the below stuff

All notable changes to this project will be documented in this file.

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

