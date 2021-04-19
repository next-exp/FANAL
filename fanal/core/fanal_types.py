import numpy as np

from enum        import Enum
from dataclasses import dataclass

import invisible_cities.core.system_of_units as units



# Deprecated
@dataclass
class VolumeDim:
	z_min : float
	z_max : float
	rad   : float



@dataclass
class BBAnalysisParams:

    buffer_Eth         : float  = np.nan
    trans_diff         : float  = np.nan
    long_diff          : float  = np.nan
    fwhm               : float  = np.nan
    e_min              : float  = np.nan
    e_max              : float  = np.nan

    procedure          : str    = ""
    voxel_size_x       : float  = np.nan
    voxel_size_y       : float  = np.nan
    voxel_size_z       : float  = np.nan
    strict_voxel_size  : bool   = False
    barycenter         : bool   = True
    voxel_Eth          : float  = np.nan

    veto_width         : float  = np.nan
    veto_Eth           : float  = np.nan

    contiguity         : float  = np.nan
    track_Eth          : float  = np.nan
    max_num_tracks     : int    = -1
    blob_radius        : float  = np.nan
    blob_Eth           : float  = np.nan

    roi_Emin           : float  = np.nan
    roi_Emax           : float  = np.nan


    def __post_init__(self):
        # Assertions
        assert self.e_max >= self.e_min,       \
            "energy_filter settings not valid: 'e_max' must be higher than 'e_min'"
        assert self.roi_Emax >= self.roi_Emin, \
            "roi_filter settings not valid: 'roi_Emax' must be higher than 'roi_Emin'"
        assert self.procedure in ['paolina_ic', 'paolina_2'], \
            "Only valid procedures are: 'paolina_ic' and 'paolina_2'"


    def set_units(self):
        "Setting units to parameters when being read from json config file"
        self.buffer_Eth   = self.buffer_Eth   * units.keV
        self.trans_diff   = self.trans_diff   * (units.mm / units.cm**.5)
        self.long_diff    = self.long_diff    * (units.mm / units.cm**.5)
        self.fwhm         = self.fwhm         * units.perCent
        self.e_min        = self.e_min        * units.keV
        self.e_max        = self.e_max        * units.keV
        self.voxel_size_x = self.voxel_size_x * units.mm
        self.voxel_size_y = self.voxel_size_y * units.mm
        self.voxel_size_z = self.voxel_size_z * units.mm
        self.voxel_Eth    = self.voxel_Eth    * units.keV
        self.veto_width   = self.veto_width   * units.mm
        self.veto_Eth     = self.veto_Eth     * units.keV
        self.contiguity   = self.contiguity   * units.mm
        self.track_Eth    = self.track_Eth    * units.keV
        self.blob_radius  = self.blob_radius  * units.mm
        self.blob_Eth     = self.blob_Eth     * units.keV
        self.roi_Emin     = self.roi_Emin     * units.keV
        self.roi_Emax     = self.roi_Emax     * units.keV


    def __repr__(self):
        s  = f"*** Buffer energy th.: {self.buffer_Eth / units.keV:.1f} keV\n"
        s += f"*** Transverse   diff: {self.trans_diff / (units.mm/units.cm**0.5):.2f}  mm/cm**0.5\n"
        s += f"*** Longitudinal diff: {self.long_diff / (units.mm/units.cm**0.5):.2f}  mm/cm**0.5\n"
        s += f"*** Energy Resolution: {self.fwhm / units.perCent:.2f}% fwhm at Qbb\n"
        s += f"*** Recons. procedure: {self.procedure}\n"
        s += f"*** Voxel Size:        ({self.voxel_size_x / units.mm}, "
        s += f"{self.voxel_size_y / units.mm}, {self.voxel_size_z / units.mm}) mm  "
        if self.procedure == "paolina_ic":
            s += f"-  strict: {self.strict_voxel_size}\n"
        else:
            s += f"-  barycenter: {self.barycenter}\n"
        s += f"*** Voxel energy th.:  {self.voxel_Eth / units.keV:.1f} keV\n"
        if self.procedure == "paolina_2":
            s += f"*** Contiguity      :  {self.contiguity / units.mm} mm\n"
        s += f"*** Track energy th.:  {self.track_Eth / units.keV:.1f} keV\n"
        s += f"*** Max num Tracks:    {self.max_num_tracks}\n"
        s += f"*** Blob radius:       {self.blob_radius:.1f} mm\n"
        s += f"*** Blob energy th.:   {self.blob_Eth/units.keV:4.1f} keV\n"
        s += f"*** ROI energy limits: ({self.roi_Emin/units.keV:4.1f}, "
        s += f"{self.roi_Emax/units.keV:4.1f}) keV\n"
        return s

    __str__ = __repr__



@dataclass
class KrAnalysisParams:

    veto_width          : float = np.nan
    tracking_sns_pde    : float = 1.
    tracking_mask_att   : float = 0.
    tracking_charge_th  : float = 0.
    correction_map_type : str   = ''

    def set_units(self):
        "Setting units to parameters when being read from json config file"
        self.veto_width = self.veto_width * units.mm

    def __repr__(self):
        s  = f"*** Veto width:            {self.veto_width / units.mm:.1f} mm\n"
        s += f"*** Tracking sensors pde:  {self.tracking_sns_pde:.3f}\n"
        s += f"*** Tracking mask att:     {self.tracking_mask_att:.3f}\n"
        s += f"*** Tracking charge th:    {self.tracking_charge_th:.3f} pes\n"
        s += f"*** Correction map type:   {self.correction_map_type}\n"
        return s

    __str__ = __repr__
