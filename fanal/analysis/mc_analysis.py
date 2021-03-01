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
# TODO: Get it from detector
S1_Eth    = 20. * units.keV  # Energy threshold for S1s
S1_WIDTH  = 10. * units.ns   # S1 time width
EVT_WIDTH =  5. * units.ms   # Recorded time per event

# Recorded time per event in S1_width units
#evt_width_inS1 = EVT_WIDTH / S1_WIDTH


# The logger
logger = get_logger('Fanal')



def check_mc_data(mcHits     : pd.DataFrame,
                  buffer_Eth : float,
                  e_min      : float,
                  e_max      : float
                 ) -> bool:
    """
    Checks if an event accomplish for:
    * mc ACTIVE energy in limits
    * No signal in BUFFER
    * Just one S1

    Parameters:
    -----------
    mcHits     : pd.DataFrame
        Dataframe containing the MC hits
    buffer_Eth : float
        Buffer energy threshold.
    e_min      : float
        Minimum MC energy
    e_max      : float
        Maximum MC energy

    Returns:
    --------
    Boolean
    """

    # Check ACTIVE energy
    active_mcHits = mcHits[mcHits.label == 'ACTIVE']
    active_energy = 0.
    if len(active_mcHits):
        active_energy = active_mcHits.energy.sum()

    # Check BUFFER energy
    buffer_mcHits = mcHits[mcHits.label == 'BUFFER']
    buffer_energy = 0.
    if len(buffer_mcHits):
        buffer_energy = buffer_mcHits.energy.sum()

    # Check number of S1s
    num_S1 = get_num_S1(active_mcHits)

    # result
    mc_filter = ( (e_min <= active_energy <= e_max) and
                  (buffer_energy <= buffer_Eth)     and
                  (num_S1 == 1) )

    # Verbosity
    logger.info(f"ACTIVE mcE: {active_energy/units.keV:.1f} keV   " + \
                f"BUFFER mcE: {buffer_energy/units.keV:.1f} keV   " + \
                f"Num S1: {num_S1}  ->  MC filter: {mc_filter}")

    # Return filtering result
    return mc_filter



def get_num_S1(evt_hits: pd.DataFrame
            ) -> int:
    """
    It computes the number of S1s with energy higher than threshold
    inside one event recorded time

    Parameters:
    -----------
    evt_hits : pd.DataFrame
        Dataframe containing the ACTIVE MC hits
    Returns:
    --------
    int : Number of S1s
    """

    num_s1 = 0

    # Discarding hits beyond the one event-width
    evt_t0 = evt_hits.time.min()
    evt_hits = evt_hits[evt_hits.time < evt_t0 + EVT_WIDTH].copy()

    # Discretizing hit times in s1_widths
    evt_hits['disc_time'] = (evt_hits.time // S1_WIDTH) * S1_WIDTH

    # Getting num S1s
    s1_times = evt_hits.disc_time.unique()
    for s1_time in s1_times:
        s1_energy = evt_hits[evt_hits.disc_time==s1_time].energy.sum()
        if s1_energy > S1_Eth: num_s1 += 1

    return num_s1



