# Development & Release Workflow

## Table of Contents

- [Development \& Release Workflow](#development--release-workflow)
  - [Table of Contents](#table-of-contents)
  - [1. Local setup](#1-local-setup)
  - [2. Making changes](#2-making-changes)
    - [Semantic versioning](#semantic-versioning)
  - [3. Running notebooks for manual exploration](#3-running-notebooks-for-manual-exploration)
    - [Local JupyterLab](#local-jupyterlab)
    - [Google Colab](#google-colab)
  - [4. Release process](#4-release-process)
  - [5. Troubleshooting](#5-troubleshooting)
    - [`uv run pytest` → "Failed to spawn: `pytest`"](#uv-run-pytest--failed-to-spawn-pytest)
    - ["ModuleNotFoundError: sklearn" during tests](#modulenotfounderror-sklearn-during-tests)

This document describes **how to work on `dcurves` day-to-day**: running the
unit-tests, validating changes locally, and cutting a release that is
published automatically by the **uv-based GitHub Actions pipeline**.

---

## 1. Local setup

```bash
# clone and enter the repo
$ git clone https://github.com/MSKCC-Epi-Bio/dcurves.git
$ cd dcurves

# create / recreate the virtual-env managed by uv
$ uv venv              # creates .venv with the current Python

# install the package **with** dev-extras (tests + lint + notebooks)
$ uv pip install -e ".[dev]"
```

You can now execute any command inside the environment with `uv run …`, e.g.

```bash
$ uv run pytest            # run the entire test-suite
```

> **Tip** – If you prefer activating the venv in your shell:
> `source .venv/bin/activate` then simply `pytest`.

## 2. Making changes

1.  Create a branch:
    `git checkout -b feature/your-topic`
2.  Modify the code **and add / update tests** under `tests/`.
3.  Run the test-suite
    `uv run pytest -q`         # fast feedback
4.  Update `CHANGELOG.md` (keep to *Keep-a-Changelog* style).
5.  Push and open a pull-request.

### Semantic versioning
* **PATCH** : bug-fixes, test-only changes.
* **MINOR** : new backwards-compatible features.
* **MAJOR** : breaking API changes.

Bump `project.version` in `pyproject.toml` accordingly **inside the PR** so the
CI can publish the correct version when the tag is pushed.

---

## 3. Running notebooks for manual exploration

### Local JupyterLab

Just open .ipynb file in vscode/cursor or do `uv run jupyter lab <filename>`

The editable install means `import dcurves` reflects live code.

### Google Colab

1. Push your branch to GitHub.
2. In Colab: `pip install git+https://github.com/<you>/dcurves@feature/your-topic`.
3. Experiment freely.

---

## 4. Release process

Releases are fully automated via **`.github/workflows/uv-actions.yml`**.

1. Merge the PR into `main`.
2. Create a signed tag matching the new version, e.g.
   git tag -s v1.2.0 -m "v1.2.0"
   git push origin v1.2.0
3. GitHub Actions matrix will:
   * run the test-suite on Python 3.9 → 3.12;
   * build wheels & source dist;
   * upload to **Test PyPI** first – only if that version isn't there;
   * on tagged commit (refs/tags/v*) upload to **PyPI**.

Tokens are stored in the repo secrets `TEST_PYPI_API_TOKEN` and `PYPI_API_TOKEN`.

---

## 5. Troubleshooting

### `uv run pytest` → "Failed to spawn: `pytest`"

The venv does not contain pytest. Run:

uv pip install -e ".[dev]"

or `uv add pytest pytest-mock`.

### "ModuleNotFoundError: sklearn" during tests

Install dev-extras which include *scikit-learn* (see above).

---
