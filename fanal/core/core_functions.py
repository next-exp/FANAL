import time
from   datetime import datetime
import numpy as np
from   typing      import Tuple, List, Iterable
from   numpy      import pi

NN = np.nan

def timeit(f):
    """
    Decorator for function timing.
    """
    def time_f(*args, **kwargs):
        t0 = time.time()
        output = f(*args, **kwargs)
        print("Time spent in {}: {} s".format(f.__name__,
                                              time.time() - t0))
        return output
    return time_f


def in_range(data, minval=-np.inf, maxval=np.inf):
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


def find_nearest(array : np.array, value :float)->float:
    """Return the array element nearest to value"""
    idx = (np.abs(array-value)).argmin()
    return array[idx]
