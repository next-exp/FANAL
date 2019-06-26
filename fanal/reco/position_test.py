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
from   invisible_cities.evm.event_model      import MCHit, Voxel

from fanal.core.fanal_types import DetName
from fanal.reco.position    import translate_hit_positions
from fanal.reco.position    import check_event_fiduciality



def test_translate_hit_positions():
    # Check of assymetric detector
    mcHits = [{'label': 'test', 'time':  0 * units.mus, 'x': 0., 'y': 0., 'z': 0., 'E': 0.01},
              {'label': 'test', 'time': 10 * units.mus, 'x': 0., 'y': 0., 'z': 0., 'E': 0.01},
              {'label': 'test', 'time': 20 * units.mus, 'x': 0., 'y': 0., 'z': 0., 'E': 0.01}]
    mcHits = pd.DataFrame(mcHits, columns = ('label', 'time', 'x', 'y', 'z', 'E'))

    drift_velocity = 1. * units.mm / units.mus
    translate_hit_positions(DetName.next100, mcHits, drift_velocity)

    OK_shifted_z = [0., 10., 20.]

    assert_array_equal(OK_shifted_z, mcHits.shifted_z)

    # Check of ssymetric detectors
    mcHits = [{'label': 'test', 'time':  0 * units.mus, 'x': 0., 'y': 0., 'z':    0., 'E': 0.01},
              {'label': 'test', 'time': 40 * units.mus, 'x': 0., 'y': 0., 'z':  500., 'E': 0.01},
              {'label': 'test', 'time': 20 * units.mus, 'x': 0., 'y': 0., 'z': -500., 'E': 0.01}]
    mcHits = pd.DataFrame(mcHits, columns = ('label', 'time', 'x', 'y', 'z', 'E'))

    drift_velocity = 1. * units.mm / units.mus
    translate_hit_positions(DetName.next500, mcHits, drift_velocity)

    OK_shifted_z = [0., 460., -480.]

    assert_array_equal(OK_shifted_z, mcHits.shifted_z)



def test_check_event_fiduciality():
    ### General data for checks of assymetric detectors
    detname    = DetName.next100
    veto_width = 20. * units.mm
    min_VetoE  =  3. * units.keV

    # Check 1: All voxels inside the fiducial volume
    voxels = [Voxel(x = 20., y = 20., z =  30., E =  5. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 30., y = 30., z = 150., E = 10. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 40., y = 30., z =  80., E = 20. * units.keV, size = (10., 10., 10.))]
    minZ, maxZ, maxRad = 30., 150., 50.
    vetoE = 0.
    fiducial = True
    assert (minZ, maxZ, maxRad, vetoE, fiducial) == \
           check_event_fiduciality(detname, veto_width, min_VetoE, voxels)

    # Check 2: Energy in veto lower than threshold
    voxels = [Voxel(x = 20., y = 20., z =   5., E =  1. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 30., y = 30., z = 150., E = 10. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 40., y = 30., z =  80., E = 20. * units.keV, size = (10., 10., 10.))]
    minZ, maxZ, maxRad = 5., 150., 50.
    vetoE = 1. * units.keV
    fiducial = True
    assert (minZ, maxZ, maxRad, vetoE, fiducial) == \
           check_event_fiduciality(detname, veto_width, min_VetoE, voxels)

    # Check 3: Energy in veto higher than threshold
    voxels = [Voxel(x = 20., y = 20., z =  5., E = 17. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 30., y = 30., z = 50., E = 10. * units.keV, size = (10., 10., 10.)),
              Voxel(x = 40., y = 30., z = 95., E = 20. * units.keV, size = (10., 10., 10.))]
    minZ, maxZ, maxRad = 5., 95., 50.
    vetoE = 17. * units.keV
    fiducial = False
    assert (minZ, maxZ, maxRad, vetoE, fiducial) == \
           check_event_fiduciality(detname, veto_width, min_VetoE, voxels)


    ### General data for checks of assymetric detectors
    detname    = DetName.next500
    veto_width = 20. * units.mm
    min_VetoE  =  3. * units.keV

    # Check 4: All voxels inside the fiducial volume
    voxels = [Voxel(x = -200., y = 100., z = -30., E =  5. * units.keV, size = (10., 10., 10.)),
              Voxel(x =    0., y = 500., z = 150., E = 10. * units.keV, size = (10., 10., 10.)),
              Voxel(x =   40., y = 130., z =  80., E = 20. * units.keV, size = (10., 10., 10.))]
    minZ, maxZ, maxRad = -30., 150., 500.
    vetoE = 0.
    fiducial = True
    assert (minZ, maxZ, maxRad, vetoE, fiducial) == \
           check_event_fiduciality(detname, veto_width, min_VetoE, voxels)

    # Check 5: Energy in central veto higher than threshold
    voxels = [Voxel(x = 120., y = 220., z =  -5., E = 17. * units.keV, size = (10., 10., 10.)),
              Voxel(x = -30., y = -30., z = 180., E = 10. * units.keV, size = (10., 10., 10.)),
              Voxel(x =   0., y = 500., z = -95., E = 20. * units.keV, size = (10., 10., 10.))]
    minZ, maxZ, maxRad = -95., 180., 500.
    vetoE = 17. * units.keV
    fiducial = False
    assert (minZ, maxZ, maxRad, vetoE, fiducial) == \
           check_event_fiduciality(detname, veto_width, min_VetoE, voxels)




