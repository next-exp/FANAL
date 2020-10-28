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



############################  POSITION FUNCTIONS  ############################

def translate_hit_positions(detname        : DetName,
                            active_mcHits  : pd.DataFrame,
                            drift_velocity : float
                           ) -> None:
    """
    In MC simulations all the hits of a MC event are assigned to the same event.
    In some special cases these events may contain hits in a period of time
    much longer than the corresponding to an event.
    Some of them occur with a delay that make them being reconstructed in shifted-Zs.
    This functions accomplish all these situations.
    """

    # Minimum time offset to fix
    min_offset = 1.*units.mus

    hit_times = active_mcHits.time
    min_time, max_time = hit_times.min(), hit_times.max()
    
    # Only applying to events wider than a minimum offset
    if ((max_time - min_time) > min_offset):
        if is_detector_symmetric(detname):
            active_mcHits['shifted_z'] = active_mcHits['z'] - np.sign(active_mcHits['z']) * \
                                         (active_mcHits['time'] - min_time) * drift_velocity
        # In case of assymetric detector
        else:
            active_mcHits['shifted_z'] = active_mcHits['z'] + \
                                         (active_mcHits['time'] - min_time) * drift_velocity

    # Event time width smaller than minimum offset        
    else:
        active_mcHits['shifted_z'] = active_mcHits['z']



def check_event_fiduciality(detname      : DetName,
                            veto_width   : float,
                            min_VetoE    : float,
                            event_voxels : List[Voxel]
                           ) -> Tuple[float, float, float, float, bool]:
    """
    Checks if an event is fiducial or not.
    
    Parameters:
    -----------
    detname      : DetName
        Name of the detector.
    veto_width   : float
        Width of the veto region.
    min_VetoE    : float
        Veto energy threshold.
    event_voxels : List[Voxel]
        List of voxels of the event.
        
    Returns:
    --------
    Tuple of 5 values containing:
        3 floats containing voxels minZ, maxZ and maxRad
        A float with the nergy deposited in veto
        A bool saying if the event is fiducial or not.
    """
    
    voxels_Z   = [voxel.Z for voxel in event_voxels]
    voxels_Rad = [(math.sqrt(voxel.X**2 + voxel.Y**2)) for voxel in event_voxels]

    voxels_minZ   = min(voxels_Z)
    voxels_maxZ   = max(voxels_Z)
    voxels_maxRad = max(voxels_Rad)

    fid_dimensions = get_fiducial_size(detname, veto_width)
    
    if is_detector_symmetric(detname):
        veto_energy = sum(voxel.E for voxel in event_voxels \
                          if ((voxel.Z < fid_dimensions.z_min) |
                              (voxel.Z > fid_dimensions.z_max) |
                              (math.sqrt(voxel.X**2 + voxel.Y**2) > fid_dimensions.rad) |
                              # Extra veto for central cathode
                              (abs(voxel.Z) < veto_width)))
    else:
        veto_energy = sum(voxel.E for voxel in event_voxels \
                          if ((voxel.Z < fid_dimensions.z_min) |
                              (voxel.Z > fid_dimensions.z_max) |
                              (math.sqrt(voxel.X**2 + voxel.Y**2) > fid_dimensions.rad)))
        
    fiducial_filter = veto_energy < min_VetoE

    return voxels_minZ, voxels_maxZ, voxels_maxRad, veto_energy, fiducial_filter



############################   ENERGY FUNCTIONS   ############################

# Some constants needed
S1_Eth    = 20. * units.keV  # Energy threshold for S1s
S1_WIDTH  = 10. * units.ns   # S1 time width
EVT_WIDTH =  5. * units.ms   # Recorded time per event

# Recorded time per event in S1_width units
evt_width_inS1 = EVT_WIDTH / S1_WIDTH



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



def smear_evt_energy(mcE       : float,
                     sigma_Qbb : float,
                     Qbb       : float
					) -> float:
	"""
	It smears and returns the montecarlo energy of the event (mcE) according to:
	the sigma at Qbb (sigma_Qbb) in absolute values (keV).
	"""
	sigma_E = sigma_Qbb * math.sqrt(mcE/Qbb)
	smE = np.random.normal(mcE, sigma_E)
	return smE
