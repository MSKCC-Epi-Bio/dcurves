#!/usr/bin/env bash
# Bash script used to install dcurves package and then run examples.py

pip -q install  ../

./start_python.sh -m unittest -s ../

#python3 -m unittest ../

