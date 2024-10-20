import numpy as np
import pandas as pd

from typing import List, Optional, Union
from itertools import combinations

class FeatureCombiner:
    __supported_operations = ['multiplication', 'sum']

    @classmethod
    def verify_fit(cls,
                    X: np.ndarray | pd.DataFrame, 
                    cols: Optional[List[str, int]]):
        # make sure they are the same type
        if cols is None:
            return 

        if isinstance(X, np.ndarray) and any(isinstance(x, str) for x in cols):
            raise TypeError("Passing string columns with a numpy array !!. Either pass a pandas dataframe or a list of indices")

        # make sure the column are of the same type
        if not all([isinstance(x, type(cols[0])) for x in cols]):
            raise TypeError("Make sure all the columns are of the same type !!")

        if not isinstance(cols[0], (str, int)):
            raise TypeError(f"Make sure the column types are either {str} or {int}") 

    def __init__(self):
        self.is_fit = False 
        # save the passed features if any
        self.features_ = None    
        # save the type of the last fit
        self.last_fit_type_ = None
        # save the number of features in case no features were passed
        self.num_features_: List[Union[int, str]] = None
        # save the number of subsets
        self.subsets_: List[int]= None
        # save the operation performed on each of the subsets
        self.op: str = None

    def _extract_features_fit(self, X: np.ndarray | pd.DataFrame, 
                         cols: Optional[List[str, int]]) -> pd.DataFrame:
        # first step is verify the input
        self.verify_fit(X, cols)
        # save the features
        self.features_ = cols

        # save the type
        self.last_fit_type_ = type(X)

        # extract as follows: 
        if cols is None:
            if isinstance(X, np.ndarray):
                return pd.DataFrame(X)
            return X

        if isinstance(X, pd.DataFrame):
            if isinstance(cols[0], str):
                return X.loc[:, cols]
            return X.iloc[:, cols]

        # wrap it in a dataframe anyway
        return pd.DataFrame(X[:, cols])

    def _extract_features_transform(self, 
                                    X: np.ndarray | pd.DataFrame, 
                                    cols: Optional[List[str, int]]) -> pd.DataFrame:
        if cols is None:
            if isinstance(X, np.ndarray):
                return pd.DataFrame(X)
            return X

        if isinstance(X, pd.DataFrame):
            if isinstance(cols[0], str):
                return X.loc[:, cols]
            return X.iloc[:, cols]

        # wrap it in a dataframe anyway
        return pd.DataFrame(X[:, cols])

    def fit(self, 
            X: np.ndarray | pd.DataFrame, 
            cols: List[Union[str, int]],
            subsets: Union[int, List[int], None],
            operation: str
            ):

        if operation not in self.__supported_operations:
            raise NotImplementedError(f"The current implementation supports only the following operations: {self.__supported_operations}")
        
        X_num = self._extract_features_fit(X, continuous_cols=cols)
        self.num_features_ = len(X_num.shape[1])

        if isinstance(subsets, int):
            subsets = [subsets]

        if subsets is None:
            subsets = list(range(2, self.num_features_))

        # make sure the number of size of each subset is at least 2
        for size in subsets:
            if size < 2:
                raise ValueError(f"The subset sizes are expected to be at least 2. Found: {size}")

        self.subsets_ = subsets


    def _generate_subsets(self, subset_size: int) -> List[int]:
        return combinations(list(range(0, ), self.num_features_), subset_size)

    def multiply_columns(self, X_input, subset: List[int]) -> pd.DataFrame | np.ndarray:
        X = X_input if isinstance(X_input, np.ndarray) else X_input.values
        X = np.prod(X[:, subset], axis=1, keepdims=True)
        # convert back to dataframe
        if isinstance(X_input, pd.DataFrame):
            return pd.DataFrame(X, X_input.index, columns=[f"feat_multiplication{'_'.join(subset)}"])

        return X

    def new_df(self, X_input):
        for subset_size in self.subsets_:
            res = [self.multiply_columns(X_input, subset=s) for s in self._generate_subsets(subset_size=subset_size)]
            if isinstance(X_input, pd.DataFrame):
                X_temp = pd.concat(res,axis=1)
                X_input = pd.concat([X_input, X_temp], axis=1)
            else:
                X_temp = np.concatenate(res, axis=1)
                X_input = np.concatenate([X_input, X_temp], axis=1)

        return X_input

    def transform(self, 
                  X: np.ndarray | pd.DataFrame,
                  cols: List[Union[str, int]]):
        if not isinstance(X, self.last_fit_type_):
            raise TypeError(f"the {str(self.__class__)} object was fit on a {self.last_fit_type_} object. Found: {type(X)}")

        if cols != self.features_:
            raise ValueError(f"Pass the same features as in the fit phase. Expected: {self.features_}. Found: {cols}")

        X_num = self._extract_features_transform(X, cols)        

        return self.new_df(X_num)

    def fit_transform(self,
            X: np.ndarray | pd.DataFrame, 
            cols: List[Union[str, int]],
            subsets: Union[int, List[int], None],
            operation: str
            ):
        self.fit(X, cols, subsets, operation)
        return self.transform(X)
