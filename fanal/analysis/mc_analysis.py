import math
import numpy as np
import pandas as pd

from   typing import Tuple
from   typing import List

# IC importings
import invisible_cities.core.system_of_units        as units

# FANAL importings
from fanal.utils.logger      import get_logger
from fanal.core.fanal_units  import Qbb

from fanal.core.detectors    import Detector
from fanal.core.detectors    import S1_Eth
from fanal.core.detectors    import S1_WIDTH
from fanal.core.detectors    import EVT_WIDTH

# The logger
logger = get_logger('Fanal')



def check_mc_data(mcHits     : pd.DataFrame,
                  buffer_Eth : float,
                  e_min      : float,
                  e_max      : float
                 ) -> Tuple[float, bool]:
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
    float  : MC energy
    Boolean: MC filter
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
    return (active_energy, mc_filter)



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



def reconstruct_hits(mcHits   : pd.DataFrame,
                     evt_mcE  : float,
                     fwhm     : float
                    ) -> pd.DataFrame:
    
    # Departing from MC hits
    recons_hits = mcHits.copy()

    # Smearing the energy: first the event energy and then
    # the hits accordingly
    sigma_at_Qbb = fwhm * Qbb / 2.355
    sigma_at_mcE = sigma_at_Qbb * math.sqrt(evt_mcE / Qbb)
    evt_smE      = np.random.normal(evt_mcE, sigma_at_mcE)
    sm_factor    = evt_smE / evt_mcE
    recons_hits.energy = recons_hits.energy * sm_factor

    # TODO: Smear positions according to diffussion

    # TODO: Translate Z position according to DRIFT velocity
#        # Translating hit Z positions from delayed hits
#        translate_hit_positions(detector, active_mcHits)
#        active_mcHits = active_mcHits[(active_mcHits.shifted_z < ACTIVE_dimensions.z_max) &
#                                      (active_mcHits.shifted_z > ACTIVE_dimensions.z_min)]
    #recons_hits = translate_hits(detname, recons_hits)


    return recons_hits



def translate_hits(detector : Detector,
                   mcHits   : pd.DataFrame
                  ) -> None:
    """
    In MC simulations all the hits of a MC event are assigned to the same event.
    In some special cases these events may contain hits in a period of time
    much longer than the corresponding to an event.
    Some of them occur with a delay that make them being reconstructed in shifted-Zs.
    This functions accomplish all these situations.
    """

    # Minimum time offset to fix
    min_offset = 1. * units.mus

    min_time, max_time = mcHits.time.min(), mcHits.time.max()

    # Only applying to events wider than a minimum offset
    if ((max_time - min_time) > min_offset):
        if detector.symmetric:
            active_mcHits['shifted_z'] = active_mcHits['z'] - np.sign(active_mcHits['z']) * \
                                         (active_mcHits['time'] - min_time) * drift_velocity
        # In case of assymetric detector
        else:
            active_mcHits['shifted_z'] = active_mcHits['z'] + \
                                         (active_mcHits['time'] - min_time) * drift_velocity

    # Event time width smaller than minimum offset
    else:
        active_mcHits['shifted_z'] = active_mcHits['z']


