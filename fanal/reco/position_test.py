"""
Tests for position
"""

import numpy  as np
import pandas as pd
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
from invisible_cities.evm.event_model        import MCHit, Voxel

from fanal.core.fanal_types import SpatialDef, VolumeDim

from fanal.reco.position import get_voxel_size
from fanal.reco.position import translate_hit_positions
from fanal.reco.position import check_event_fiduciality



def test_get_voxel_size():
    with pytest.raises(AttributeError):
        x = get_voxel_size(SpatialDef.non_existing)

    size = get_voxel_size(SpatialDef.low)
    assert type(size) == tuple
    assert len(size)  == 3



def test_translate_hit_positions():
    mcHits = [{'label': 'test', 'time':  0 * units.mus, 'x': 0., 'y': 0., 'z': 0., 'E': 0.01},
              {'label': 'test', 'time': 10 * units.mus, 'x': 0., 'y': 0., 'z': 0., 'E': 0.01},
              {'label': 'test', 'time': 20 * units.mus, 'x': 0., 'y': 0., 'z': 0., 'E': 0.01}]
    mcHits = pd.DataFrame(mcHits, columns = ('label', 'time', 'x', 'y', 'z', 'E'))

    drift_velocity = 1. * units.mm / units.mus
    translate_hit_positions(mcHits, drift_velocity)

    OK_shifted_z = [0., 10., 20.]

    assert_array_equal(OK_shifted_z, mcHits.shifted_z)



def test_check_event_fiduciality():
    fid_dimensions = VolumeDim(z_min = 10. * units.mm,
                               z_max = 90. * units.mm,
                               rad   = 90. * units.mm)
    min_VetoE = 8. * units.keV

    # Check 1: All voxels inside the fiducial volume
    Voxels = [Voxel(x = 20., y = 20., z = 20., E =  5. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 30., y = 30., z = 50., E = 10. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 40., y = 30., z = 80., E = 20. * units.keV, size = (10., 10., 10.))]
    minZ, maxZ, maxRad = 20., 80., 50.
    vetoE = 0.
    fiducial = True
    assert (minZ, maxZ, maxRad, vetoE, fiducial) == \
           check_event_fiduciality(Voxels, fid_dimensions, min_VetoE)

    # Check 2: Energy in veto lower than threshold
    Voxels = [Voxel(x = 20., y = 20., z =  5., E =  5. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 30., y = 30., z = 50., E = 10. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 40., y = 30., z = 80., E = 20. * units.keV, size = (10., 10., 10.))]
    minZ, maxZ, maxRad = 5., 80., 50.
    vetoE = 5. * units.keV
    fiducial = True
    assert (minZ, maxZ, maxRad, vetoE, fiducial) == \
           check_event_fiduciality(Voxels, fid_dimensions, min_VetoE)

    # Check 3: Energy in veto higher than threshold
    Voxels = [Voxel(x = 20., y = 20., z = 20., E =  5. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 30., y = 30., z = 50., E = 10. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 40., y = 30., z = 95., E = 20. * units.keV, size = (10., 10., 10.))]
    minZ, maxZ, maxRad = 20., 95., 50.
    vetoE = 20. * units.keV
    fiducial = False
    assert (minZ, maxZ, maxRad, vetoE, fiducial) == \
           check_event_fiduciality(Voxels, fid_dimensions, min_VetoE)
