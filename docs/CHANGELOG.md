# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.2] - 2024-12-15 [YANKED]
- No issues with tests or GitHub actions, but major issues in pulling from PyPI
- Have to investigate


### Added
- Support for Python 3.13.
- Compatibility updates for Black formatting tool.

### Changed
- Updated project configuration to include Python 3.13 testing environment.
- Removed unnecessary dependencies and imports.
- Fixed broken link in project description.

### Fixed
- Resolved configuration issues for Python 3.13 compatibility.
- Updated tox configuration and project dependencies.

## [1.1.1] - 2024-10-25

### Added
- Support for Python 3.12.
- New test data utilities and expanded testing:
  - Plot saving, smoothing, and data loading tests.
- Mocking setup for matplotlib in test files for improved test stability.

### Changed
- Refined `smooth_frac` validation in plot functions, enforcing a valid range of 0 to 1.
- Plotting code updated to assign specific colors to "all" and "none" models, with predefined colors for up to four additional models.
- Enhanced error handling for plot data, including validation for empty DataFrames and unmatched model names in smoothed data.
- Configuration updates:
  - Added Python 3.12 to `tox.ini` and GitHub Actions.
  - Revised `tox` and `pytest` warnings filters for deprecated functions in numpy, matplotlib, and other dependencies.
- Documentation reorganized:
  - Moved all changelogs to `/docs` and removed obsolete files related to `mkdocs`.
  - Updated instructions in `README.md` and removed legacy `towncrier` references.

### Fixed
- Added error handling for empty DataFrames in `dca`, preventing potential plotting issues.
- Updated `dca` function to raise errors for invalid inputs, ensuring robustness in model handling.
- Resolved runtime and deprecation warnings across test files, streamlining future Python and package compatibility.

## [Released]

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
- Improved GitHub Actions workflow: lint only `dcurves/` directory, run pylint before pytest.

### Removed
- Support for Python 3.8.

## [1.0.6.5] - 2023-09-12

### Added
- Merged split_dca with main branch. (GH-6)
- New site for tutorial, replacing Dan's site. (GH-14)
- `show_grid` option for plots. (GH-21)

### Changed
- Updated `README.md`. (GH-15)
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
