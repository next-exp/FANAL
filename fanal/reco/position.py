import math
import numpy as np

from typing import Tuple

# Specific IC stuff
from invisible_cities.core.system_of_units_c  import units

# Specific fanal stuff
from fanal.core.fanal_types import SpatialDef



def get_voxel_size(spatial_def: SpatialDef) -> Tuple[float, float, float]:
	"""
	It returns the size of voxels to use.
	spatial_def: SpatialDef: enum ('low' or 'high')
	"""

	if spatial_def == SpatialDef.low:
		return (10., 10., 5.)

	elif spatial_def == SpatialDef.high:
		return (2., 2., 2.)



def translate_hit_positions(mcHits, drift_velocity):
	"""
	In MC simulations all the hits of a MC event are assigned to the same event.
	In some special cases these events may contain hits in a period of time
	much longer than the corresponding to an event.
	Some of them occur with a delay that make them being reconstructed in shifted-Zs.
	This functions accomplish all these situations.
	"""

	mc_times = np.array([hit.time for hit in mcHits])
	min_time, max_time = min(mc_times), max(mc_times)

	# Only applying to events wider than 1 micro second
	if ((max_time-min_time) > 1.*units.mus):
		transPositions = [(hit.X, hit.Y, hit.Z + (hit.time-min_time) * drift_velocity) \
			for hit in mcHits]
	else:
		transPositions = [(hit.X, hit.Y, hit.Z) for hit in mcHits]
        
	return transPositions



def check_event_fiduciality(event_voxels, fid_minZ, fid_maxZ, fid_maxRad, min_VetoE):
	"""
	"""

	voxels_Z = [voxel.Z for voxel in event_voxels]
	voxels_Rad = [(math.sqrt(voxel.X**2 + voxel.Y**2)) for voxel in event_voxels]

	voxels_minZ = min(voxels_Z)
	voxels_maxZ = max(voxels_Z)
	voxels_maxRad = max(voxels_Rad)

	fiducial_filter = ((voxels_minZ > fid_minZ) & (voxels_maxZ < fid_maxZ) &
					   (voxels_maxRad < fid_maxRad))

	veto_energy = 0.
                
	# If there is any voxel in veto, checking if their energies are higher than threshold
	if not fiducial_filter:
		veto_energy = sum(voxel.E for voxel in event_voxels if ((voxel.Z < fid_minZ) |
			(voxel.Z < fid_maxZ) | (math.sqrt(voxel.X**2 + voxel.Y**2) > fid_maxRad)))
        
	if veto_energy < min_VetoE:
		fiducial_filter = True
        
	return voxels_minZ, voxels_maxZ, voxels_maxRad, veto_energy, fiducial_filter

