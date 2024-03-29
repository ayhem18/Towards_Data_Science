"""
This script implements a number of utility functions
"""
import random
import numpy as np 

from typing import Sequence

def _find_index(x: float, 
               sorted_list: Sequence[np.number], 
               min_index: int, 
               max_index: int) -> int:
    # let's start with the base cases
    if x == sorted_list[min_index]: 
        return min_index
    
    if x < sorted_list[min_index]:
        return min_index - 1
    
    if x >= sorted_list[max_index]:
        return max_index

    if max_index - min_index <= 1:
        return min_index

    mid_index = (max_index + min_index) // 2

    if x == sorted_list[mid_index]:
        return mid_index

    # at this point, we have already established that x > sorted_list[min_index]
    # and x < sorted_list[max_index]
    if x < sorted_list[mid_index]:
        return _find_index(x, sorted_list, min_index=min_index, max_index=mid_index)
    
    return _find_index(x, sorted_list, min_index=mid_index, max_index=max_index)

def find_index(x: float, 
               sorted_list: Sequence[np.number]) -> int:
    return _find_index(x, sorted_list, 0, len(sorted_list) - 1)


def find_index_linear(x: float, 
                      sorted_list: Sequence[np.number],
                      ) -> int:
    if x < sorted_list[0]:
        return -1
    
    if x >= sorted_list[-1]:
        return len(sorted_list) - 1

    i = 0
    while sorted_list[i] < x:
        i += 1
    
    if sorted_list[i] == x:
        return i
    return i - 1


def main(): 
    for _ in range(100):
        n = random.randint(20, 100)
        l = sorted(np.unique([random.randint(0, 1000) for _ in range(n)]).tolist())
        for j in range(100):
            x = random.randint(0, 100)
            i1 = find_index_linear(x, l)
            i2 = find_index(x, l)
            assert i1 == i2, f"found different indices: linear: {l[i1]}, divide & conquer: {l[i2]}"

if __name__ == '__main__':
    main()
