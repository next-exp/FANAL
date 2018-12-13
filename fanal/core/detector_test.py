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

from hypothesis					 import given, settings
from hypothesis.strategies 		 import integers
from hypothesis.strategies 		 import floats
from fanal.core.fanal_exceptions import DetectorNameNotDefined
from fanal.core.fanal_types      import DetName
from fanal.core.fanal_types      import ActiveVolumeDim
from fanal.core.detector         import get_active_size

def test_get_active_size():
	for det in [DetName.new, DetName.next100, DetName.next500]:
		assert type(get_active_size(det)) == ActiveVolumeDim

	try:
		x = get_active_size('unknown')
	except DetectorNameNotDefined:
		pass
