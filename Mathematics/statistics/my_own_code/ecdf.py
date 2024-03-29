"""
This script contains a simple implementation of the Emperical CDF
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, List, Union

from utils import find_index

class ECDF:    
    def __init__(self, 
                 data: Sequence[np.number]) -> None:
        # make sure the data is uni-dimensional
        if isinstance(data, np.ndarray):
            if data.squeeze().ndim > 1:
                raise ValueError(f"The current implementation of the ECDF class only supports one-dimensional data")
            # make sure to squeeze the numpy array
            data = data.squeeze()

        if isinstance(data, List):
            if len(data) == 1 and isinstance(data[0], Sequence):
                data = data[0]
            elif len(data) != 1 and isinstance(data[0], Sequence):
                raise ValueError(f"The current implementation of the ECDF class only supports one-dimensional data")

        self.data = np.sort(np.asarray(data))
        
    def _cdf(self, x: float) -> float: 
        """This function estimates the probability P(X <= x): which is basically f(x) where f is CDF of 'X'
        Returns:
            float: P(X <= x)
        """ 
        # first of all find where the new point falls with respect to the data
        i = find_index(x, self.data)
        
        # we consider special cases, if the new value is less than the minimum sample value
        if i == -1:
            temp =  (x / self.data[0]) * (1 / (len(self.data) + 1))
            return max(0, temp)

        if i == len(self.data) - 1:
            temp = (len(self.data) / (len(self.data) + 1)) + (1 - self.data[-1] / x ) * (1 / (len(self.data) + 1))
            return min(temp, 1)
        
        return ((i + 1) + (x - self.data[i]) / (self.data[i + 1] - self.data[i])) / (len(self.data) + 1)

    def cdf(self, x: Union[float, Sequence[float]] = None) -> List[float]:
        if x is None: 
            x = self.data.copy()

        if isinstance(x, (float, int)):
            x = [x]
        
        return [round(self._cdf(val), 5) for val in x]

    def __call__(self, x: Union[float, Sequence[float]] = None) -> float:
        """this function
        Returns:
            float: _description_
        """
        return self.cdf(x)


if __name__ == '__main__':
    x = [-5, -2, 1, 4, 6, 10, 23, 45, 50, 100, 2, 56, 68, 90, 95]
    xs = [-1] + x + [250]
    e = ECDF(x)
    ys = e(xs)

    for i1, i2 in zip(xs, ys):
        print(f'{i1}: {i2}')
    
    # plt.plot(x + [110], ys)
    # plt.title("emperical cdf")
    # plt.xlabel('x')
    # plt.ylabel('P(X <= x)')
    # plt.yticks(ys)
    # plt.xticks(sorted(x + [110]))
    # plt.show()


