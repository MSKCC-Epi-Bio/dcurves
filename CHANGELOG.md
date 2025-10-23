# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.6] - 2025-10-23

### Added
- Support for Python 3.13 (all 89 tests pass)
- Session-scoped pytest fixtures in conftest.py for all test data files
- Python version classifiers (3.9-3.13) to pyproject.toml

### Changed
- Updated GitHub Actions to test on Python 3.13
- Centralized test data loading via fixtures (18 session-scoped fixtures)
- Updated test files to use conftest fixtures instead of direct load_data() calls
- Improved test efficiency through fixture reuse

### Removed
- Redundant runtime dependencies: typing, setuptools, scipy (indirect via lifelines/statsmodels)
- Redundant dev dependency: pathlib (built-in since Python 3.4)
- Duplicate test setup code in test_binary_consequences.py
- Redundant test files: test_load_test_data.py, test_non_dcurves_code.py (4 tests)
- Duplicate plotting test files: test_optional_gridlines.py, test_plot_saving.py (5 tests)
- Split large test_plotting.py into test_plotting_core.py and test_plotting_options.py for clarity
- Dead code: _validate.py (195 lines of commented validation functions)
- Dead code: load_test_data.py from main package (moved to tests/load_test_data.py)
- Commented code block in dca.py (18 lines of unused statistical calculations)
- Result: 78 tests (from 89) in 13 files (from 16), -263 lines of dead code from package

## [1.1.5] - 2025-07-11

### Changed
- Moved `images/` directory to `public/` for better project organization.
- Removed `.tox` and `.idea` directories from version control and updated `.gitignore`.

### Fixed
- Updated statsmodels dependency from `==0.14.4` to `>=0.14.5` to fix compatibility with scipy>=1.14.0. The issue was caused by statsmodels using scipy's private `_lazywhere` function which was removed in scipy 1.14.0. This fix ensures dcurves works with the latest scipy versions.
- Fixed missing data files in package distribution. Added `dcurves/data/*.csv` to package includes to ensure test data files are properly distributed with the package.

## [1.1.4] - 2025-05-22

### Added
* `linewidths` parameter in plotting (`plot_graphs`, `_plot_net_benefit`, `_plot_net_intervention_avoided`).  Accepts either a single float applied to all models or a per-model list.
* Comprehensive unit-tests for linewidth handling and updated marker/colour logic.
* `docs/WORKFLOW.md` – end-to-end contributor guide.

### Changed
* `VALID_MARKERS` de-duplicated and centralised.
* README expanded with badges, table-of-contents and clearer examples.
* Jupyter notebooks folder renamed to `notebooks/`.
* CI workflow (`uv-actions.yml`) simplified; dev dependencies installed via `.[dev]`.
* Dynamic version retrieval in `dcurves/__init__.py` via `importlib.metadata` (removes hard-coded version string).
* Development dependencies consolidated under `[project.optional-dependencies].dev`.

### Removed
* Legacy files: `.pylintrc`, `.coverage`, `poetry.lock`.

## [1.1.3] - 2025-03-15

- **Added:**
  - Black and white (bw) support for plotting to enhance visual clarity
  - pdoc action to automatically generate the documentation site for the library
- **Changed:**
  - Switched from Poetry to UV to manage dependencies
  - CI/CD pipelines updated via GitHub Actions for testing, documentation generation, and package publishing using UV
  - Project metadata, documentation, and changelog updated to reflect broader improvements
- **Removed:**
  - Black formatting tool along with unnecessary dependencies and redundant directories

## [1.1.1] - 2024-10-25

- **Added:**
  - Support for Python 3.12.
  - New test data utilities and expanded testing:
    - Plot saving, smoothing, and data loading tests.
  - Mocking setup for matplotlib in test files for improved test stability.
- **Changed:**
  - Refined `smooth_frac` validation in plot functions, enforcing a valid range of 0 to 1.
  - Plotting code updated to assign specific colors to "all" and "none" models, with predefined colors for up to four additional models.
  - Enhanced error handling for plot data, including validation for empty DataFrames and unmatched model names in smoothed data.
  - Configuration updates:
    - Added Python 3.12 to `tox.ini` and GitHub Actions.
    - Revised `tox` and `pytest` warnings filters for deprecated functions in numpy, matplotlib, and other dependencies.
  - Documentation reorganized:
    - Moved all changelogs to `/docs` and removed obsolete files related to `mkdocs`.
    - Updated instructions in `README.md` and removed legacy `towncrier` references.
- **Fixed:**
  - Added error handling for empty DataFrames in `dca`, preventing potential plotting issues.
  - Updated `dca` function to raise errors for invalid inputs, ensuring robustness in model handling.
  - Resolved runtime and deprecation warnings across test files, streamlining future Python and package compatibility.

## [1.1.0] - 2024-02-09

- **Added:**
  - Support for Python 3.11.
  - Lowess smoothing functionality with the `smooth_frac` parameter for plot functions.
  - Options to display legend and grid in plots.
  - Plot quality and saving settings for enhanced user control over plot outputs.
- **Changed:**
  - Updated to support the latest Pandas version.
  - Enhanced data validation with new checks and corresponding unit tests.
  - Updated docstrings across various modules for better clarity.
  - Improved GitHub Actions workflow: lint only `dcurves/` directory and run pylint before pytest.
- **Removed:**
  - Support for Python 3.8.

## [1.0.6.5] - 2023-09-12

- **Added:**
  - Merged split_dca with the main branch. (GH-6)
  - New site for tutorial, replacing Dan's site. (GH-14)
  - `show_grid` option for plots. (GH-21)
- **Changed:**
  - Updated `README.md`. (GH-15)
  - Improved data loading and cross-validation processes. (GH-21)
  - Achieved 100% test coverage. (GH-18)
- **Fixed:**
  - Resolved tox issues in actions. (GH-13)
  - Fixed GitHub actions and release notes system. (GH-20)

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

- **Added:**
  - Initial release of the Decision Curve Analysis plotting module.
  - Core functionality for creating net benefit and net intervention avoided plots.

## [0.0.3] - 2022-06-01

## [0.0.2] - 2022-05-01

- Pre-release versions. Development and testing.

## [Yanked]

## [1.1.2] - 2024-12-15 [YANKED]

- No issues with tests or GitHub actions, but major issues in pulling from PyPI.
- Have to investigate.
- **Added:**
  - Support for Python 3.13.
  - Compatibility updates for Black formatting tool.
- **Changed:**
  - Updated project configuration to include Python 3.13 testing environment.
  - Removed unnecessary dependencies and imports.
  - Fixed broken link in project description.
- **Fixed:**
  - Resolved configuration issues for Python 3.13 compatibility.
  - Updated tox configuration and project dependencies.
