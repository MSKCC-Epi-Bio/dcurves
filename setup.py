from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.0.2.4'
DESCRIPTION = 'Python package for Andrew Vickers\' Decision Curve Analysis method to evaluate prediction models and diagnostic tests'
LONG_DESCRIPTION = 'A package that allows to build simple streams of video, audio and camera data.'

# Setting up
setup(
    name="dcurves",
    version=VERSION,
    author="Shaun Porwal",
    author_email="<shaun.porwal@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'statsmodels', 'lifelines', 'matplotlib'],
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