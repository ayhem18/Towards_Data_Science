"""
This script contains simple functionalities to discretize continuous features
"""

import numpy as np
import pandas as pd

from typing import List, Optional
from sklearn.cluster import KMeans


class KMeansDiscretizer:
    @classmethod
    def verify_fit(cls,
                    X: np.ndarray | pd.DataFrame, 
                    continuous_cols: Optional[List[str, int]]):
        # make sure they are the same type
        if continuous_cols is None:
            return 

        if isinstance(X, np.ndarray) and any(isinstance(x, str) for x in continuous_cols):
            raise TypeError("Passing string columns with a numpy array !!. Either pass a pandas dataframe or a list of indices")

        # make sure the column are of the same type
        if not all([isinstance(x, type(continuous_cols[0])) for x in continuous_cols]):
            raise TypeError("Make sure all the columns are of the same type !!")

        if not isinstance(continuous_cols[0], (str, int)):
            raise TypeError(f"Make sure the column types are either {str} or {int}") 

    
    def __init__(self):
        self.is_fit = False 
        # save the fitted KMeans objects
        self.k_means_ = []

        # save the passed features if any
        self.features_ = None    
        # save the type of the last fit
        self.last_fit_type_ = None
        # save the number of features in case no features were passed
        self.num_features_ = None
        # save the number of clusters in the 'fit' stage
        self.num_clusters = None

    def extract_features(self, X: np.ndarray | pd.DataFrame, continuous_cols: Optional[List[str, int]]) -> pd.DataFrame:
        # first step is verify the input
        self.verify_fit(X, continuous_cols)
        # save the features
        self.features = continuous_cols

        # save the type
        self.last_fit_type = type(X)

        # extract as follows: 
        if continuous_cols is None:
            return X

        if isinstance(X, pd.DataFrame):
            if isinstance(continuous_cols[0], str):
                return X.loc[:, continuous_cols]
            return X.iloc[:, continuous_cols]

        # wrap it in a dataframe anyway
        return pd.DataFrame(X[:, continuous_cols])


    def fit(self, X: np.ndarray | pd.DataFrame, 
            continuous_cols: Optional[List[str, int]], 
            num_clusters: int):

        X_num = self.extract_features(X, continuous_cols=continuous_cols)
        
        
        self.num_features = len(X_num.shape[1])
        self.num_clusters = num_clusters

        # TODO: try to use threading and
        for col_index, col in enumerate(X_num.columns): 
            self.k_means_.append(KMeans(n_clusters=num_clusters, random_state=0))
            self.k_means_[col_index].fit(X_num.loc[:, [col]])
            

    def _verify_fit(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:

        
        pass

    def transform(self, X: np.ndarray | pd.DataFrame):
        pass
