import sys
import time

import pandas as pd
import numpy  as np

from typing   import Tuple, List, Any



def is_interactive() -> bool:
    """
    It returns if the code is being run interactively
    """
    import __main__ as main
    return not hasattr(main, '__file__')



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



def bin_data_with_equal_bin_size(data     : List[np.array],
                                 bin_size : List[float]
                                ) -> Tuple[List[np.array], List[float]]:
    """
    Given a list of numpy arrays (or any sequence with max() and min())
    and   a list of bin sizes return:
    * arrays with the bin edges
    * floats with effective bin sizes
    """
    assert len(data) == len(bin_size), "ERROR: Different dimensions for data and bin_size"
    bin_data      = []
    bin_eff_sizes = []
    for i in range(len(data)):
        x = data[i]
        size = bin_size[i]
        width        = x.max() - x.min()
        nbins        = int(width / size) + 1
        bin_eff_size = width / nbins
        xbins        = np.histogram_bin_edges(x, nbins)
        bin_data     .append(xbins)
        bin_eff_sizes.append(bin_eff_size)
    return bin_data, bin_eff_sizes



def get_barycenter(positions : np.array,
                   weights   : np.array) -> np.array:
    """Computes baricenter as the product of position positions and weights"""

    if (np.sum(weights) == 0.):
        print("WARNING: trying to get barycenter with all-zero weights.")
        return np.mean(positions)

    return np.dot(positions, weights) / np.sum(weights)



def get_size(obj  : Any,
             seen : bool = False
            )    -> None:
    """
    Recursively finds object´s size

    parameters:
    obj : Object to be sized

    returns:
    size : Object size in bytes
    """

    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])

    return size



def my_value_counts(photons    : pd.DataFrame,
                    value_name : str
                   )          -> None :

    total   = len(photons)
    entries = photons[value_name].value_counts()
    for key in entries.keys():
        print(f"{key:20}: {entries[key]:10}  ({entries[key] * 100 / total:.5}%)")

