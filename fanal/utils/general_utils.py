import time
import math

import numpy as np

from datetime import datetime
from typing   import Tuple, List, Iterable
from numpy    import pi

NN = np.nan



def timeit(f):
    """
    Decorator for function timing.
    """
    def time_f(*args, **kwargs):
        t0 = time.time()
        output = f(*args, **kwargs)
        print(f"Time spent in {f.__name__}: {time.time() - t0} s")
        return output
    return time_f



def in_range(data, minval=-np.inf, maxval=np.inf) :
    """
    Find values in range [minval, maxval).

    Parameters
    ---------
    data : np.ndarray
        Data set of arbitrary dimension.
    minval : int or float, optional
        Range minimum. Defaults to -inf.
    maxval : int or float, optional
        Range maximum. Defaults to +inf.

    Returns
    -------
    selection : np.ndarray
        Boolean array with the same dimension as the input. Contains True
        for those values of data in the input range and False for the others.
    """
    return (minval <= data) & (data < maxval)



def find_nearest(array : np.array, value :float) -> float:
    """
    Return the array element nearest to value
    """
    idx = (np.abs(array-value)).argmin()
    return array[idx]



def distance(pos1 : np.array, pos2 : np.array) -> float :
    """
    Returns the distance between the 2 positions
    """
    return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2 + (pos1[2]-pos2[2])**2)

