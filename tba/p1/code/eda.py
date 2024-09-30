import os
import pandas as pd, numpy as np, sqlite3 as sq

from typing import Union, List, Tuple
from pathlib import Path 
from sklearn.model_selection import train_test_split

def missing_data_analysis(df: pd.DataFrame) -> pd.DataFrame:
    return df.mean(df.isna(), axis=1)
