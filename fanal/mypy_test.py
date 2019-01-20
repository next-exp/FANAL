"""
General test suite to type_check with mypy all the sw
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



os.chdir(os.path.expandvars('$FANALPATH/fanal'))


def test_core_tye_checker():
	mypy_result = ('', '', 0)
	assert  mypy_result == api.run(['core'])


def test_reco_tye_checker():
	mypy_result = ('', '', 0)
	assert  mypy_result == api.run(['reco'])


def test_ana_tye_checker():
	mypy_result = ('', '', 0)
	assert  mypy_result == api.run(['ana'])


def test_fanal_reco_tye_checker():
	mypy_result = ('', '', 0)
	assert  mypy_result == api.run(['fanal_reco.py'])


def test_fanal_ana_tye_checker():
	mypy_result = ('', '', 0)
	assert  mypy_result == api.run(['fanal_ana.py'])
