import math
import numpy as np
import pandas as pd

from   typing import Tuple
from   typing import List

# IC importings
import invisible_cities.core.system_of_units        as units
from invisible_cities.evm.event_model        import Voxel  as icVoxel

# FANAL importings
from fanal.utils.logger      import get_logger
from fanal.core.fanal_units  import Qbb

from fanal.core.detectors    import Detector
from fanal.core.detectors    import S1_Eth
from fanal.core.detectors    import S1_WIDTH
from fanal.core.detectors    import EVT_WIDTH

# The logger
logger = get_logger('Fanal')



# FANAL importings
from fanal.core.fanal_types  import DetName
from fanal.core.detector     import get_fiducial_size
from fanal.core.detector     import is_detector_symmetric
from fanal.core.fanal_units  import drift_velocity



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
