"""
General test suite to type_check with mypy all the sw
"""

import os
import numpy as np

from pytest                import mark
from pytest                import approx
from pytest                import raises
from flaky                 import flaky
from numpy.testing         import assert_array_equal
from numpy.testing         import assert_allclose

from hypothesis            import given, settings
from hypothesis.strategies import integers
from hypothesis.strategies import floats

from mypy import api



os.chdir(os.path.expandvars('$FANALPATH/fanal'))


def test_core_tye_checker():
	assert  api.run(['core'])[2] == 0


def test_reco_tye_checker():
	assert  api.run(['reco'])[2] == 0


def test_ana_tye_checker():
	assert  api.run(['ana'])[2] == 0


#def test_mc_tye_checker():
#	assert  api.run(['mc'])[2] == 0


def test_fanal_reco_tye_checker():
	assert  api.run(['fanal_reco.py'])[2] == 0


def test_fanal_ana_tye_checker():
	assert  api.run(['fanal_ana.py'])[2] == 0
