import os
import pandas as pd, numpy as np, sqlite3 as sq

from typing import Union, List, Tuple
from pathlib import Path 
from sklearn.model_selection import train_test_split

def missing_data_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return df.mean(df.isna(), axis=1)

# develop some mechanisms to determine whether to keep the outliers or discard them.

def iqr_outliers(df: pd.DataFrame, 
                 column: str | int, 
                 add_column:bool=False) -> pd.Series:
    
    if isinstance(column, int):
        col = df.columns.tolist()[column]
    else:
        col = column
    
    q1, q3 = np.quantile(df[col], 0.25), np.quantile(df[col], 0.75)
    iqr = q3 - q1
    min_val, max_val = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    # https://pandas.pydata.org/docs/reference/api/pandas.Series.between.html
    mask = df[col].between(min_val, max_val)

    if add_column:
        df['is_outlier'] = ~mask

    return mask
