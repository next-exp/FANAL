"""
Tests for core_functions
"""

import numpy as np

from pytest        import mark
from pytest        import approx
from pytest        import raises
from flaky         import flaky
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from hypothesis            import given, settings
from hypothesis.strategies import integers
from hypothesis.strategies import floats

from invisible_cities.core.testing_utils import exactly
from invisible_cities.core.testing_utils import float_arrays
from invisible_cities.core.testing_utils import FLOAT_ARRAY
from invisible_cities.core.testing_utils import random_length_float_arrays

from fanal.core.core_functions import find_nearest


def nearest(a, v):
    """Alternative (not optimized) implementation of find_nearest
    Used for testing purpose only
    """
    nr =a[0]
    diff = 1e+9
    for x in a:
        if np.abs(x-v) < diff:
            nr = x
            diff = np.abs(x-v)
    return nr


def test_simple_find_nearest():
    x = np.arange(100)
    assert find_nearest(x, 75.6) == exactly(76)
    assert find_nearest(x, 75.5) == exactly(75)


def test_gauss_find_nearest():
    e = np.random.normal(100, 10, 100)

    for x in range(1, 100, 10):
        assert find_nearest(e, x)   == approx(nearest(e, x), rel=1e-3)


@given(float_arrays(min_value=1,
                    max_value=100))
def test_find_nearest(data):
    assert find_nearest(data, 10) == approx(nearest(data, 10), rel=1e-3)
