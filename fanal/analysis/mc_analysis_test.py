import pytest
import os

from invisible_cities.io.mcinfo_io  import load_mchits_df

from fanal.analysis.mc_analysis     import check_mc_data
from fanal.analysis.mc_analysis     import get_num_S1
from fanal.analysis.mc_analysis     import smear_hit_energies
from fanal.analysis.mc_analysis     import smear_hit_positions
from fanal.analysis.mc_analysis     import translate_hit_zs


@pytest.fixture(scope='module')
def asymmetric_mcHits(FANAL_DATA_DIR):
    fname  = os.path.join(FANAL_DATA_DIR, 'next100/Bi214/sim/next100.Bi214.000.next.h5')
    mcHits = load_mchits_df(fname)
    return mcHits.loc[0]


#@pytest.fixture(scope='module')
#def symmetric_mcHits(FANAL_DATA_DIR):
#    fname  = os.path.join(FANAL_DATA_DIR, 'next_hd/bb0nu/sim/next100.bb0nu.000.next.h5')
#    mcHits   = load_mchits_df()
#    return mcHits
#
#
#@pytest.fixture(scope  = 'module',
#                params = ["asymmetric_mcHits", "symmetric_mcHits"],
#                ids    = ["Asymmetric hits", "Symmetric hits"])
#def mcHits(request):
#    return request.getfixturevalue(request.param)


def test_check_mc_data():
    # TODO
    pass


def test_get_num_S1():
    # TODO
    pass


def test_smear_hit_energies():
    # TODO
    pass


def test_smear_hit_positions():
    # TODO
    pass


def test_translate_hit_zs(asymmetric_mcHits):
    # TODO
    pass

