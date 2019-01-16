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

from fanal.reco.reco_io_functions import get_reco_group_name
from fanal.core.fanal_types       import SpatialDef


def test_raise_value_error_if_unknown_SpatialDef():
	with pytest.raises(AttributeError):
		x = get_reco_group_name(0.5, SpatialDef.does_not_exist)


def test_reco_group_name():
	fwhm = 0.5
	spd  = SpatialDef.low
	reco_group_name = f'/FANALIC/RECO_05fmhm_lowDef'
	assert reco_group_name == get_reco_group_name(fwhm, spd)


