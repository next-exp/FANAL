"""
Tests for ana_io_functions
"""

import numpy as np
import pytest
from pytest        import mark
from pytest        import approx
from pytest        import raises
from flaky         import flaky
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from hypothesis            import given, settings
from hypothesis.strategies import integers
from hypothesis.strategies import floats
from invisible_cities.core.testing_utils       import exactly
from invisible_cities.core.testing_utils       import float_arrays
from invisible_cities.core.testing_utils       import FLOAT_ARRAY
from invisible_cities.core.testing_utils       import random_length_float_arrays

from . ana_io_functions     import get_ana_group_name
from fanal.core.fanal_types import SpatialDef as spd

def test_simple_ana_group_name():
    fwhm = 0.5
    sigma_a = 1
    ana_group_name = f'/FANALIC/ANA_05fmhm_lowDef'
    assert ana_group_name == get_ana_group_name(fwhm, spd.low)


def test_raise_value_error_if_unknown_enum():
    with pytest.raises(AttributeError):
        x = get_ana_group_name(0.5, spd.does_not_exist)
