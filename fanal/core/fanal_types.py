import numpy as np

from typing import Tuple
from typing import Dict
from typing import List
from typing import TypeVar
from typing import Optional

from enum        import Enum
from dataclasses import dataclass

import invisible_cities.core.system_of_units               as units



# Deprecated
@dataclass
class VolumeDim:
	z_min : float
	z_max : float
	rad   : float



@dataclass
class AnalysisParams:
    trans_diff         : float  = np.nan
    long_diff          : float  = np.nan
    fwhm               : float  = np.nan
    e_min              : float  = np.nan
    e_max              : float  = np.nan

    voxel_size_x       : float  = np.nan
    voxel_size_y       : float  = np.nan
    voxel_size_z       : float  = np.nan
    strict_voxel_size  : bool   = False
    voxel_Eth          : float  = np.nan

    veto_width         : float  = np.nan
    veto_Eth           : float  = np.nan

    track_Eth          : float  = np.nan
    max_num_tracks     : int    = -1
    blob_radius        : float  = np.nan
    blob_Eth           : float  = np.nan

    roi_Emin           : float  = np.nan
    roi_Emax           : float  = np.nan

    def __repr__(self):
        s  = f"*** Transverse   diff: {self.trans_diff / (units.mm/units.cm**0.5):.2f}  mm/cm**0.5\n"
        s += f"*** Longitudinal diff: {self.long_diff / (units.mm/units.cm**0.5):.2f}  mm/cm**0.5\n"
        s += f"*** Energy Resolution: {self.fwhm / units.perCent:.2f}% fwhm at Qbb\n"
        s += f"*** Voxel Size:        ({self.voxel_size_x / units.mm}, "
        s += f"{self.voxel_size_y / units.mm}, {self.voxel_size_z / units.mm}) mm  "
        s += f"-  strict: {self.strict_voxel_size}\n"
        s += f"*** Voxel energy th.:  {self.voxel_Eth / units.keV:.1f} keV\n"
        s += f"*** Track energy th.:  {self.track_Eth / units.keV:.1f} keV\n"
        s += f"*** Max num Tracks:    {self.max_num_tracks}\n"
        s += f"*** Blob radius:       {self.blob_radius:.1f} mm\n"
        s += f"*** Blob energy th.:   {self.blob_Eth/units.keV:4.1f} keV\n"
        s += f"*** ROI energy limits: ({self.roi_Emin/units.keV:4.1f}, "
        s += f"{self.roi_Emax/units.keV:4.1f}) keV\n"
        return s
