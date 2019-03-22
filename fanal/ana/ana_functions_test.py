"""
Tests for ana_functions
"""

import numpy as np
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
from invisible_cities.reco.paolina_functions import make_track_graphs


from fanal.reco.reco_io_functions import get_voxels_reco_dict

from fanal.ana.ana_functions import get_new_energies
from fanal.ana.ana_functions import get_voxel_track_relations
from fanal.ana.ana_functions import process_tracks



# Voxel collection with the 5th one having negligible energy,
# and being the 2nd voxel, its closest neighbour.
voxels_dict = get_voxels_reco_dict()
voxel_Eth   = 0.005
voxels_dict['event_id'] = [ 1 ,    2 ,    3 ,    4 ,    5 ]
voxels_dict['X']        = [30.,   30.,   30.,   30.,   30.]
voxels_dict['Y']        = [30.,   30.,   30.,   30.,   40.]
voxels_dict['Z']        = [30.,   40.,   50.,   60.,   40.]
voxels_dict['E']        = [0.1,   0.1,   0.1,   0.1,   0.001]
voxels_dict['negli']    = [False, False, False, False, True]
voxels_df1 = pd.DataFrame(voxels_dict)

def test_get_new_energies():
    new_energies = [0.1, 0.101, 0.1, 0.1, 0.]
    assert new_energies == get_new_energies(voxels_df1)



# Voxel collection grouped into 3 different tracks.
# 1st track with the 2 first voxels,     and energy > track_Eth
# 2nd track with the following 1 voxel,  and energy < track_Eth
# 3rd track with the following 3 voxels, and energy > track_Eth (highest)
# And finally a negligible voxel
voxels_dict = get_voxels_reco_dict()
voxel_Eth   = 0.005
voxels_dict['event_id'] = [ 1 ,    2 ,    3 ,    4 ,    5 ,    6 ,    7 ]
voxels_dict['X']        = [30.,   30.,   60.,   60.,   60.,   60.,   20.]
voxels_dict['Y']        = [30.,   30.,   60.,   60.,   60.,   60.,   20.]
voxels_dict['Z']        = [30.,   40.,   30.,   60.,   70.,   80.,   80.]
voxels_dict['E']        = [0.1,   0.1,   0.1,   0.1,   0.1,   0.1,   0.001]
voxels_dict['negli']    = [False, False, False, False, False, False, True]
voxels_df2 = pd.DataFrame(voxels_dict)

voxel_dimensions = (10., 10., 10.)
ic_voxels = [Voxel(voxels_df2.iloc[i].X, voxels_df2.iloc[i].Y,
                   voxels_df2.iloc[i].Z,
                   voxels_df2.iloc[i].E, voxel_dimensions) \
             for i in range(len(voxels_df2))]

event_tracks = make_track_graphs(ic_voxels)



def test_get_voxel_track_relations():
    relations = [0, 0, 1, 2, 2, 2, np.nan]
    assert relations == get_voxel_track_relations(voxels_df2, event_tracks)



def test_process_tracks():
    track_Eth = 0.15
    tracks_E  = [0.3, 0.2]
    sorted_tracks = process_tracks(event_tracks, track_Eth)

    assert tracks_E[0] == pytest.approx(sorted_tracks[0][0])
    assert tracks_E[1] == pytest.approx(sorted_tracks[1][0])

    assert event_tracks[2] == sorted_tracks[0][1]
    assert event_tracks[0] == sorted_tracks[1][1]
