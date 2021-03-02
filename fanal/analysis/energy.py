import math
import numpy as np
import pandas as pd

from   typing import Tuple
from   typing import List

# IC importings
import invisible_cities.core.system_of_units        as units

# FANAL importings
from   fanal.utils.logger    import get_logger


# Some constants needed
S1_Eth    = 20. * units.keV  # Energy threshold for S1s
S1_WIDTH  = 10. * units.ns   # S1 time width
EVT_WIDTH =  5. * units.ms   # Recorded time per event

# Recorded time per event in S1_width units
evt_width_inS1 = EVT_WIDTH / S1_WIDTH


# The logger
logger = get_logger('Fanal')



def get_S1_E(evt_hits: pd.DataFrame
            ) -> List[Tuple[float,float]]:
    """
    It returns a List of tuples containing:
    (S1 time, S1 energy) for all the S1s
    inside the recorded_time per event
    """
    # Discretizing hit times in s1_widths
    evt_t0 = evt_hits.time.min()
    evt_hits['disc_time'] = ((evt_hits['time'] - evt_t0) // S1_WIDTH) * S1_WIDTH

    # Getting S1s Info
    S1s = []

    s1_ts = evt_hits[evt_hits.disc_time <= evt_width_inS1]['disc_time'].unique()
    for s1_t in s1_ts:
        s1_E = evt_hits[evt_hits.disc_time==s1_t].energy.sum()
        S1s.append((s1_t, s1_E))

    return S1s



def get_mc_energy(evt_hits: pd.DataFrame
                 ) -> float:
    """
    It returns the mc energy of an event if there is only one S1.
    It returns 0 in any other case.
    """

    S1_Es   = get_S1_E(evt_hits)
    logger.debug(f"  Num S1s:  {len(S1_Es)}")
    logger.debug(f"  S1 collection (S1_t, S1_E):  {S1_Es}")

    # Checking the number of S1s with E > Eth,
    # and collecting the energy
    num_S1s = 0
    evt_mcE = 0.
    for i in range(len(S1_Es)):
        S1_E = S1_Es[i][1]
        if S1_E > S1_Eth:
            num_S1s += 1
            if (num_S1s == 1):
                evt_mcE = S1_E
            # If more than 1 S1 with E > Eth -> Set evt_mcE to 0
            else:
                evt_mcE = 0.
                break
        # Collecting the energy of S1 with E < Eth
        else:
            evt_mcE += S1_E

    return evt_mcE



def smear_evt_energy(mcE  : float,
                     fwhm : float
                    ) -> float:
    """
    It smears and returns the montecarlo energy of the event (mcE) according to:
    the sigma at Qbb (sigma_Qbb) in absolute values (keV).
    """
    sigma_Qbb = fwhm * Qbb / 2.355
    sigma_E = sigma_Qbb * math.sqrt(mcE/Qbb)
    smE = np.random.normal(mcE, sigma_E)
    return smE
