import math
import numpy  as np
import pandas as pd

from typing import Tuple, List

# Specific IC stuff
import invisible_cities.core.system_of_units as units
from invisible_cities.evm.event_model         import MCHit, Voxel

# Specific fanal stuff
from fanal.core.fanal_types import SpatialDef, VolumeDim



def get_voxel_size(spatial_def : SpatialDef) -> Tuple[float, float, float]:
    """
    It returns a tuple with the size of voxels to use.
    spatial_def: SpatialDef: enum ('low' or 'high')
    """
    if spatial_def == SpatialDef.low:
        size = (15., 15., 15.)

    elif spatial_def == SpatialDef.std:
        size = (10., 10., 10.)

    elif spatial_def == SpatialDef.high:
        size = (3., 3., 3.)

    return size



def translate_hit_positions(active_mcHits : pd.DataFrame,
                            drift_velocity: float) -> None:
    """
    In MC simulations all the hits of a MC event are assigned to the same event.
    In some special cases these events may contain hits in a period of time
    much longer than the corresponding to an event.
    Some of them occur with a delay that make them being reconstructed in shifted-Zs.
    This functions accomplish all these situations.
    """
    
    hit_times = active_mcHits.time
    min_time, max_time = hit_times.min(), hit_times.max()
    
    # Only applying to events wider than 1 micro second
    if ((max_time - min_time) > 1.*units.mus):
        active_mcHits['shifted_z'] = active_mcHits['z'] + \
                                     (active_mcHits['time'] - min_time) * drift_velocity
    else:
        active_mcHits['shifted_z'] = active_mcHits['z']



def check_event_fiduciality(event_voxels   : List[Voxel],
                            fid_dimensions : VolumeDim,
							min_VetoE      : float
						   ) -> Tuple[float, float, float, float, bool]:
	"""
	Checks if an event is fiducial or notself.
	Parameters:
	-----------
	event_voxels   : List[Voxel]
		List of voxels of the event.
	fid_dimensions : VolumeDim
		Dimensions of the detector fiducial volume.
	min_VetoE      : float
		Veto energy threshold.
	Returns:
	--------
	Tuple of 5 values containing:
	 	3 floats containing voxels minZ, maxZ and maxRad
		A float with the nergy deposited in veto
		A bool saying if the event is fiducial or not.
	"""
	voxels_Z = [voxel.Z for voxel in event_voxels]
	voxels_Rad = [(math.sqrt(voxel.X**2 + voxel.Y**2)) for voxel in event_voxels]

	voxels_minZ   = min(voxels_Z)
	voxels_maxZ   = max(voxels_Z)
	voxels_maxRad = max(voxels_Rad)

	fiducial_filter = ((voxels_minZ   > fid_dimensions.z_min) &
	                   (voxels_maxZ   < fid_dimensions.z_max) &
					   (voxels_maxRad < fid_dimensions.rad))

	veto_energy = 0.

	# If there is any voxel in veto
	# checking if their energies are higher than threshold
	if not fiducial_filter:
		veto_energy = sum(voxel.E for voxel in event_voxels \
		                  if ((voxel.Z < fid_dimensions.z_min) |
			                  (voxel.Z > fid_dimensions.z_max) |
							  (math.sqrt(voxel.X**2 + voxel.Y**2) > fid_dimensions.rad)))

	if veto_energy < min_VetoE:
		fiducial_filter = True

	return voxels_minZ, voxels_maxZ, voxels_maxRad, veto_energy, fiducial_filter
