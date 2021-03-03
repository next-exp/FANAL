# General importings
import math
import numpy      as np
import pandas     as pd
from   typing import Tuple

# IC importings
import invisible_cities.core.system_of_units   as units

# FANAL importings
from fanal.utils.logger      import get_logger
from fanal.core.fanal_units  import Qbb

from fanal.core.detectors    import Detector
from fanal.core.detectors    import S1_Eth
from fanal.core.detectors    import S1_WIDTH
from fanal.core.detectors    import EVT_WIDTH
from fanal.core.detectors    import DRIFT_VELOCITY
from fanal.core.detectors    import MIN_TIME_SHIFT

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

    # Check ACTIVE mc energy
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



def reconstruct_hits(detector   : Detector,
                     mcHits     : pd.DataFrame,
                     evt_mcE    : float,
                     fwhm       : float,
                     trans_diff : float,
                     long_diff  : float
                    ) -> pd.DataFrame:
    
    # Departing from MC hits
    recons_hits = mcHits.copy()

    # Smearing the energy
    smear_hit_energies(recons_hits, evt_mcE, fwhm)

    # Smear positions according to spatial diffussion
    smear_hit_positions(recons_hits, detector, trans_diff, long_diff)

    # Translate Z position according to DRIFT velocity
    # TODO: Not sure if needed after checking just one mc S1
    translate_hit_zs(recons_hits, detector)

    return recons_hits



def smear_hit_energies(mcHits   : pd.DataFrame,
                       evt_mcE  : float,
                       fwhm     : float
                      ) -> None:
    """
    First smears the whole event energy and then
    the hits accordingly
    """
    if fwhm == 0. : return

    sigma_at_Qbb  = fwhm * Qbb / 2.355
    sigma_at_mcE  = sigma_at_Qbb * math.sqrt(evt_mcE / Qbb)
    evt_smE       = np.random.normal(evt_mcE, sigma_at_mcE)
    sm_factor     = evt_smE / evt_mcE
    mcHits.energy = mcHits.energy * sm_factor



def smear_hit_positions(mcHits     : pd.DataFrame,
                        detector   : Detector,
                        trans_diff : float,
                        long_diff  : float
                       ) -> None:
    """
    Smearing hits positions according to detector diffusion
    """
    if (trans_diff == long_diff == 0.) : return

    # Getting the drift length
    if detector.symmetric:
        drift_length = detector.active_z_max - abs(mcHits.z)
    else:
        drift_length = mcHits.z

    # Applying the smearing
    sqrt_len = drift_length ** 0.5

    mcHits.x = np.random.normal(mcHits.x, sqrt_len * trans_diff)
    mcHits.y = np.random.normal(mcHits.y, sqrt_len * trans_diff)
    mcHits.z = np.random.normal(mcHits.z, sqrt_len * long_diff)



def translate_hit_zs(mcHits   : pd.DataFrame,
                     detector : Detector
                    ) -> None:
    """
    In MC simulations all the hits of a MC event are assigned to the same event.
    In some special cases these events may contain hits in a period of time
    much longer than the corresponding to an event.
    Some of them occur with a delay that make them being reconstructed in shifted-Zs.
    This functions accomplish all these situations.
    """
    # Only applying to events wider than a minimum offset
    min_time, max_time = mcHits.time.min(), mcHits.time.max()
    if ((max_time - min_time) <= MIN_TIME_SHIFT): return

    # In case of symmetric detector
    if detector.symmetric:
        mcHits.z = mcHits.z - np.sign(mcHits.z) * \
                  (mcHits.time - min_time) * DRIFT_VELOCITY
    # In case of asymmetric detector
    else:
        mcHits.z = mcHits.z + (mcHits.time - min_time) * DRIFT_VELOCITY
