"""
Test for fanal_reco
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


def test_fanal_reco_tye_checker():
	fanal_reco_path = os.path.expandvars('$FANALPATH/fanal')
	os.chdir(fanal_reco_path)

	mypy_result = ('', '', 0)
	assert  mypy_result == api.run(['fanal_reco.py'])
