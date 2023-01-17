from dcurves.risks import *
from os import path

#useful directories
root_test_dir = path.dirname(path.realpath(__file__))
resources_dir = path.join(root_test_dir, 'resources')
simdata_dir = path.join(resources_dir, 'sim_data')

r_results_dir = path.join(resources_dir, 'r_results')




# Load data
import dcurves
from dcurves.load_test_data import load_tutorial_bin_marker_risks_list

# load risk functions
# from dcurves.risks import _calc_binary_risks

# load tools
import pandas as pd
import numpy as np



















