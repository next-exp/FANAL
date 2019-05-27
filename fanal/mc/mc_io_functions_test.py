"""
Tests for mc_io_functions
"""

import numpy  as np
import pandas as pd
import pytest

from typing        import Dict, List, Any, Tuple

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

from fanal.mc.mc_io_functions import load_mc_hits
from fanal.mc.mc_io_functions import load_mc_particles
from fanal.mc.mc_io_functions import get_num_mc_particles


iFile_name = '../data/sim/bb0nu/bb0nu-001-.next.h5'


def get_str_Nostr_data(data_list: List) -> Tuple[List[str], List[Any]]:
#def get_str_Nostr_data(data_list):
    str_data    = [data for data in data_list if isinstance(data, str)]
    NOstr_data  = [data for data in data_list if not isinstance(data, str)]
    return str_data, NOstr_data


def test_load_mc_hits():
    hits_df = load_mc_hits(iFile_name)

    # Check all the hits are read
    assert len(hits_df) == 78930

    # Check the indexes of first & last hits
    assert hits_df.index[0] == (1000, 2, 0)
    assert hits_df.index[len(hits_df)-1] == (1788, 5, 2)

    # Check the content of the first hit
    OK_hit_data = ['ACTIVE', 0.015683162957429886, -31.395719528198242,
                   -61.9436149597168, 292.9808349609375, 0.03601445257663727]
    OK_str_data, OK_NOstr_data = get_str_Nostr_data(OK_hit_data)

    hit_data = hits_df.iloc[0].tolist()
    str_data, NOstr_data = get_str_Nostr_data(hit_data)

    assert str_data == OK_str_data
    assert np.allclose(OK_NOstr_data, NOstr_data)

    # Check the content of the last hit
    OK_hit_data = ['ACTIVE', 0.36129564, 306.96472, 290.3872,
                   896.02893, 0.015498596]
    OK_str_data, OK_NOstr_data = get_str_Nostr_data(OK_hit_data)

    hit_data = hits_df.iloc[len(hits_df)-1].tolist()
    str_data, NOstr_data = get_str_Nostr_data(hit_data)

    assert str_data == OK_str_data
    assert np.allclose(OK_NOstr_data, NOstr_data)



def test_load_mc_particles():
    parts_df = load_mc_particles(iFile_name)

    # Check all the parts are read
    assert len(parts_df) == 7833

    # Check the indexes of first & last particles
    assert parts_df.index[0] == (1000, 2)
    assert parts_df.index[len(parts_df)-1] == (1788, 5)

    # Check the content of the first particle
    OK_part_data = ['e-', True, 0, -28.38556, -64.55176, 295.22992, 0.0,
                    -62.49741, -67.622055, 305.9416, 0.4950654, 'ACTIVE',
                    'ACTIVE', -1.4136, 1.22481, -1.05618, 1.696955, 'none']
    OK_str_data, OK_NOstr_data = get_str_Nostr_data(OK_part_data)

    part_data = parts_df.iloc[0].tolist()
    str_data, NOstr_data = get_str_Nostr_data(part_data)

    assert str_data == OK_str_data
    assert np.allclose(OK_NOstr_data, NOstr_data)

    # Check the content of the last particle
    OK_part_data = ['e-', False, 3, 306.91064, 290.34573, 896.08344,
                    0.36015007, 306.96472, 290.3872, 896.02893, 0.36129564,
                    'ACTIVE', 'ACTIVE', 0.07109733, 0.03666496,
                    -0.17602018, 0.03535458, 'phot']
    OK_str_data, OK_NOstr_data = get_str_Nostr_data(OK_part_data)

    part_data = parts_df.iloc[len(parts_df)-1].tolist()
    str_data, NOstr_data = get_str_Nostr_data(part_data)

    assert str_data == OK_str_data
    assert np.allclose(OK_NOstr_data, NOstr_data)



def test_get_num_mc_particles():
    file_extents = pd.read_hdf(iFile_name, 'MC/extents', mode='r')

    assert True

