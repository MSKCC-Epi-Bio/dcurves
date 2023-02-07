#!/bin/zsh

poetry run black dcurves/*
poetry run pylint dcurves/*
