"""
Test for fanal_ana
"""

import os
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

from mypy import api


def test_fanal_ana_tye_checker():
	fanal_ana_path = os.path.expandvars('$FANALPATH/fanal')
	os.chdir(fanal_ana_path)
	
	mypy_result = ('', '', 0)
	assert  mypy_result == api.run(['fanal_ana.py'])
