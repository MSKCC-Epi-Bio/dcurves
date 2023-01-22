from setuptools import setup, find_packages
import codecs
import os

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent

long_description = (this_directory / "README.md").read_text()

VERSION = '1.0.4'
DESCRIPTION = 'Python package for Andrew Vickers\' Decision Curve Analysis method to evaluate prediction models and diagnostic tests'

# Setting up
setup(
    name="dcurves",
    version=VERSION,
    author="Shaun Porwal",
    author_email="<shaun.porwal@gmail.com>",
    description=DESCRIPTION,
    long_description=(this_directory / "README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['pandas>=1.0.2', 'numpy>=1.21.6', 'beartype>=0.11.0', 'typing>=3.7.4.3',
                      'statsmodels>=0.11.1', 'lifelines>=0.26.3', 'matplotlib>=3.2.0'],
    keywords=['python', 'dcurves', 'decision', 'curve', 'analysis', 'MSK'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    include_package_data=True,
    package_data={'': ['data/*.csv']}
)

