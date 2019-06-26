"""
Tests for detector
"""

import numpy as np

from pytest        import mark
from pytest        import approx
from pytest        import raises
from flaky         import flaky
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from hypothesis									 import given, settings
from hypothesis.strategies       import integers
from hypothesis.strategies       import floats

import invisible_cities.core.system_of_units as units

from fanal.core.fanal_exceptions import DetectorNameNotDefined
from fanal.core.fanal_types      import DetName
from fanal.core.fanal_types      import VolumeDim

from fanal.core.detector         import get_active_size
from fanal.core.detector         import get_fiducial_size
from fanal.core.detector         import is_detector_symmetric



def test_get_active_size():
	for det in [DetName.new, DetName.next100, DetName.next500]:
		assert type(get_active_size(det)) == VolumeDim

	try:
		x = get_active_size('unknown')
	except DetectorNameNotDefined:
		pass



def test_get_fiducial_size():
	fid_dim = VolumeDim(z_min =   20.  * units.mm,
	                    z_max = 1280.  * units.mm,
						rad   =  514.5 * units.mm)

	assert fid_dim == get_fiducial_size(DetName.next100, 20. * units.mm)



def test_is_detector_symmetric():
	assert is_detector_symmetric(DetName.new)     == False
	assert is_detector_symmetric(DetName.next100) == False
	assert is_detector_symmetric(DetName.next500) == True
