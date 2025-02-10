
# Use this, after doing `poetry env use pythonx.xx`, to make the env into a kernel that can be used by jupyter

`poetry run python -m ipykernel install --user --name="$(poetry run python -c 'import sys, tomli; print(tomli.load(sys.stdin.buffer)["tool"]["poetry"]["name"])' < pyproject.toml)-py$(poetry run python --version | cut -d' ' -f2 | cut -d. -f1-2)" --display-name "Python ($(poetry run python -c 'import sys, tomli; print(tomli.load(sys.stdin.buffer)["tool"]["poetry"]["name"])' < pyproject.toml), Python $(poetry run python --version | cut -d' ' -f2 | cut -d. -f1-2))"`
