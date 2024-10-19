import pandas as pd
import numpy as np

from typing import List, Tuple

def classify_features(df: pd.DataFrame, 
                      max_discrete_unique_vals: int = 20,
                      min_continuous_unique_portion: float = 0.8) -> Tuple[List[str], List[str], List[str]]:  
    non_num_cols = []
    num_cols = []

    num_discrete_cols = []
    num_continuous_cols = []

    for c in df.columns:
        if np.isnan(pd.to_numeric(df[c], errors='coerce').values).sum() > 0:
            non_num_cols.append(c)
        else:
            num_cols.append(c)

    # iterate through the numerical and count the number of unqieu values
    for nc in num_cols:
        if df[nc].nunique() < max_discrete_unique_vals: 
            num_discrete_cols.append(nc)
        else:
            # there are few cases to consider here
            # if the values are floats (not only by data type but the values can actually be decimals)
            decimal_vals = len([v for v in df[nc] if int(v) != v])
            
            if decimal_vals > 0:
                # this means that there are indeed decimals values in this column and it has to be considered loat and hence continuous
                num_continuous_cols.append(nc)
                continue 

            # at this point we know that the values are integers, in this case, it is possible that we are using some so of id
            # these values are unlikely to bleong to the numerical scale but rather the nominal (not even ordinal) scale and hence should be treated as categorical variables
            # check for the total number of unique values
            if df[nc].nunique() > min_continuous_unique_portion:
                num_discrete_cols.append(nc)
            else:
                num_continuous_cols.append(nc)
    
    return non_num_cols, num_discrete_cols, num_continuous_cols

