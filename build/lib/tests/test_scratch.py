

import pandas as pd
import numpy as np

def test_scratch1():

    asdf = pd.Series([1,2,3,4,5,6,7])

    asdf.replace(to_replace=7, value=1, inplace=True)

    print(asdf.to_string())