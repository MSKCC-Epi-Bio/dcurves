#!/bin/zsh

versions=("~3.4" "~3.5" "~3.6" "~3.7" "~3.8" "~3.9" "~3.10" "~3.11.1")

for version in "${versions[@]}"; do
  echo "Python version changed to: $version"
  sed -i "s/python_version = \".*\"/python_version = \"$version\"/g" pyproject.toml
  current_version=$(grep -oP 'python_version = "\K[^"]+' pyproject.toml)
  echo "Current Python version: $current_version"
done

