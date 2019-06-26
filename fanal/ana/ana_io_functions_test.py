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

from invisible_cities.core.testing_utils import exactly
from invisible_cities.core.testing_utils import float_arrays
from invisible_cities.core.testing_utils import FLOAT_ARRAY
from invisible_cities.core.testing_utils import random_length_float_arrays

import invisible_cities.core.system_of_units  as     units
from fanal.ana.ana_io_functions               import get_ana_group_name



def test_ana_group_name():
    fwhm = 0.74
    voxel_size = (10. * units.mm, 3. * units.cm, 15.)
    ana_group_name = '/FANALIC/ANA_fwhm_074_voxel_10x30x15'
    assert ana_group_name == get_ana_group_name(fwhm, voxel_size)

