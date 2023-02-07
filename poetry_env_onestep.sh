#!/bin/zsh

poetry env remove $(poetry env list | awk '{print $1}')

rm poetry.lock

poetry lock
poetry install
poetry shell
