"""
Tests for reco_io_functions
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
from fanal.reco.reco_io_functions             import get_reco_group_name



def test_reco_group_name():
    fwhm = 0.51
    voxel_size = (12. * units.mm, .5 * units.cm, 6.)
    reco_group_name = '/FANAL/RECO_fwhm_051_voxel_12x5x6'
    assert reco_group_name == get_reco_group_name(fwhm, voxel_size)


