"""
Tests for energy
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

import invisible_cities.core.system_of_units as units
from invisible_cities.evm.event_model        import MCHit

from fanal.reco.energy import smear_evt_energy
from fanal.reco.energy import smear_hit_energies



@flaky(max_runs=10, min_passes=9)
def test_smear_evt_energy():
    Qbb   = 2457.83 * units.keV
    mc_e  = Qbb
    sigma = 10. * units.keV
    # 90% of times the smeared value must be in +- 2 sigmas ...
    assert abs(smear_evt_energy(mc_e, sigma, Qbb) - mc_e) <= 2. * sigma



def test_smear_hit_energies():
    mcHits = [MCHit(pos = (0.,0.,0.), E = 1., t = 0., l = 'test'),
              MCHit(pos = (0.,0.,0.), E = 2., t = 0., l = 'test'),
              MCHit(pos = (0.,0.,0.), E = 4., t = 0., l = 'test')]
    conv_factor = 1.1
    sm_energies = np.array([1.1, 2.2, 4.4])
    assert_array_equal(sm_energies, smear_hit_energies(mcHits, conv_factor))
