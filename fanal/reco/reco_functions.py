import math
import numpy as np
import pandas as pd

from   typing import List, Tuple, Dict

# Specific IC stuff
import invisible_cities.core.system_of_units        as units
from   invisible_cities.evm.event_model         import MCHit
from   invisible_cities.evm.event_model         import Voxel
from   invisible_cities.reco.paolina_functions  import voxelize_hits

# Specific fanal stuff
from   fanal.core.logger             import get_logger
from   fanal.core.fanal_types        import VolumeDim
from   fanal.core.fanal_types        import DetName
from   fanal.core.detector           import get_fiducial_size
from   fanal.core.detector           import is_detector_symmetric

from   fanal.reco.reco_io_functions  import get_event_reco_data
from   fanal.reco.reco_io_functions  import extend_voxels_reco_dict


### DATA NEEDED
Qbb  = 2457.83 * units.keV
DRIFT_VELOCITY = 1. * units.mm / units.mus


### The reco logger
logger = get_logger('FanalReco')



### Reconstruct event
def reconstruct_event(det_name          : DetName,
                      ACTIVE_dimensions : VolumeDim,
                      event_number      : int,
                      event_type        : str,
                      sigma_Qbb         : float,
                      e_min             : float,
                      e_max             : float,
                      voxel_size        : np.ndarray,
                      voxel_Eth         : float,
                      veto_width        : float,
                      min_veto_e        : float,
                      event_mcParts     : pd.DataFrame,
                      event_mcHits      : pd.DataFrame,
                      voxels_dict       : Dict
                     ) -> Dict:
    
    # Data to be filled
    event_data = get_event_reco_data()
    
    # Filtering hits
    active_mcHits = event_mcHits[event_mcHits.label == 'ACTIVE'].copy()

    event_data['event_id']    = event_number
    event_data['num_MCparts'] = len(event_mcParts)
    event_data['num_MChits']  = len(active_mcHits)
    
    # The event mc energy is the sum of the energy of all the hits except
    # for Bi214 events, in which the number of S1 in the event is considered
    if (event_type == 'Bi214'):
        event_data['mcE'] = get_mc_energy(active_mcHits)
    else:
        event_data['mcE'] = active_mcHits.energy.sum()
            
    # Smearing the event energy
    event_data['smE'] = smear_evt_energy(event_data['mcE'], sigma_Qbb, Qbb)

    # Applying the smE filter
    event_data['smE_filter'] = (e_min <= event_data['smE'] <= e_max)

    # Verbosing
    logger.info(f"  Num mcHits: {event_data['num_MChits']:3}   "       + \
                f"mcE: {event_data['mcE']/units.keV:.1f} "             + \
                f"keV   smE: {event_data['smE']/units.keV:.1f} keV   " + \
                f"smE_filter: {event_data['smE_filter']}")
                
    # For those events passing the smE filter:
    if event_data['smE_filter']:

        # Smearing hit energies
        smearing_factor = event_data['smE'] / event_data['mcE']
        active_mcHits['smE'] = active_mcHits['energy'] * smearing_factor

        # Translating hit Z positions from delayed hits
        translate_hit_positions(det_name, active_mcHits, DRIFT_VELOCITY)                
        active_mcHits = active_mcHits[(active_mcHits.shifted_z < ACTIVE_dimensions.z_max) &
                                      (active_mcHits.shifted_z > ACTIVE_dimensions.z_min)]

        # Creating the IChits with the smeared energies and translated Z positions
        IChits = active_mcHits.apply(lambda hit: MCHit((hit.x, hit.y, hit.shifted_z),
                                             hit.time, hit.smE, 'ACTIVE'), axis=1).tolist()

        # Voxelizing using the IChits ...
        event_voxels = voxelize_hits(IChits, voxel_size, strict_voxel_size=False)
        event_data['num_voxels'] = len(event_voxels)
        eff_voxel_size = event_voxels[0].size
        event_data['voxel_sizeX'] = eff_voxel_size[0]
        event_data['voxel_sizeY'] = eff_voxel_size[1]
        event_data['voxel_sizeZ'] = eff_voxel_size[2]
    
        # Storing voxels info
        for voxel_id in range(len(event_voxels)):
            extend_voxels_reco_dict(voxels_dict, event_number, voxel_id,
                                    event_voxels[voxel_id], voxel_Eth)
    
        # Check fiduciality
        event_data['voxels_minZ'], event_data['voxels_maxZ'], event_data['voxels_maxRad'], \
        event_data['veto_energy'], event_data['fid_filter'] = \
        check_event_fiduciality(det_name, veto_width, min_veto_e, event_voxels)
                   
        # Verbosing
        logger.info(f"  NumVoxels: {event_data['num_voxels']:3}   "             + \
                    f"minZ: {event_data['voxels_minZ']:.1f} mm   "              + \
                    f"maxZ: {event_data['voxels_maxZ']:.1f} mm   "              + \
                    f"maxR: {event_data['voxels_maxRad']:.1f} mm   "            + \
                    f"veto_E: {event_data['veto_energy']/units.keV:.1f} keV   " + \
                    f"fid_filter: {event_data['fid_filter']}")
                
        for voxel in event_voxels:
            logger.debug(f"    Voxel pos: ({voxel.X/units.mm:5.1f}, "               + \
                         f"{voxel.Y/units.mm:5.1f}, {voxel.Z/units.mm:5.1f}) mm   " + \
                         f"E: {voxel.E/units.keV:5.1f} keV")

    return event_data
