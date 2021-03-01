import math
import numpy  as np
import pandas as pd

from typing import Tuple
from typing import List

# IC importings
import invisible_cities.core.system_of_units     as units

from invisible_cities.evm.event_model        import Voxel  as icVoxel

# FANAL importings
from fanal.core.fanal_types  import DetName
from fanal.core.detector     import get_fiducial_size
from fanal.core.detector     import is_detector_symmetric
from fanal.core.fanal_units  import drift_velocity



def translate_hit_positions(detname        : DetName,
                            active_mcHits  : pd.DataFrame
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
                            veto_Eth     : float,
                            event_voxels : List[icVoxel]
                           ) -> Tuple[float, float, float, float, bool]:
    """
    Checks if an event is fiducial or not.

    Parameters:
    -----------
    detname      : DetName
        Name of the detector.
    veto_width   : float
        Width of the veto region.
    veto_Eth     : float
        Veto energy threshold.
    event_voxels : List[icVoxel]
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

    fiducial_filter = veto_energy < veto_Eth

    return voxels_minZ, voxels_maxZ, voxels_maxRad, veto_energy, fiducial_filter
