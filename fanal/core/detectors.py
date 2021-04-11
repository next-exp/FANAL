# General importings
import math

from dataclasses import dataclass

from typing      import Callable
from typing      import Sequence
from typing      import List

# IC importings
import invisible_cities.core.system_of_units               as units
from   invisible_cities.evm.event_model       import Voxel as icVoxel

# FANAL importings
from fanal.core.fanal_exceptions              import DetectorNameNotDefined


# Common constants to all detectors
# TODO: Make them detector dependant ??
S1_Eth         = 20. * units.keV  # Energy threshold for S1s
S1_WIDTH       =  1. * units.mus  # S1 time width
EVT_WIDTH      =  5. * units.ms   # Recorded time per event
DRIFT_VELOCITY =  1. * units.mm / units.mus
MIN_TIME_SHIFT =  1. * units.mus


@dataclass(frozen=True)
class Detector:
    name               : str
    active_z_min       : float
    active_z_max       : float
    active_rad         : float
    energy_sensors     : List[str]
    tracking_sensors   : List[str]
    sensor_ids         : dict
    tracking_sns_pde   : float = 1.
    tracking_mask_att  : float = 0.
    tracking_charge_th : float = 0.

    def __post_init__(self):
        #self.symmetric : bool = (-self.active_z_min == self.active_z_max)
        # Defined this way and not the other one to be compatible with frozen=True
        object.__setattr__(self, "symmetric", (-self.active_z_min == self.active_z_max))

        # assert all energy and tracking sensors have defined ids
        assert sorted(self.energy_sensors + self.tracking_sensors) == \
               sorted(self.sensor_ids.keys()), "Mismatch on sensor definitions"

    def __repr__(self):
        s  = f"* Detector name: {self.name}\n"
        s += f"  ACTIVE dimensions (mm): Zmin = {self.active_z_min / units.mm:.2f} "
        s += f"  Zmax = {self.active_z_max / units.mm:.2f} "
        s += f"  Rad = {self.active_rad / units.mm:.2f}\n"
        s += f"  Symmetric: {self.symmetric}\n"
        s += f"  Energy   Sensors: {self.energy_sensors}\n"
        s += f"  Tracking Sensors: {self.tracking_sensors}\n"
        if (self.tracking_sns_pde != 1.):
            s += f"  Tracking Sensors PDE: {self.tracking_sns_pde}\n"
        if (self.tracking_mask_att != 0.):
            s += f"  Tracking Mask Attenuation: {self.tracking_mask_att}\n"
        s += f"  Tracking Charge Threshold: {self.tracking_charge_th} pes\n"
        return s

    __str__ = __repr__

    def get_fiducial_checker(self, veto_width : float) -> Callable :
        fiduc_z_min = self.active_z_min + veto_width
        fiduc_z_max = self.active_z_max - veto_width
        fiduc_rad   = self.active_rad   - veto_width

        if self.symmetric:
            def fiducial_checker(voxel : icVoxel):
                return not ((voxel.Z < fiduc_z_min) |
                            (voxel.Z > fiduc_z_max) |
                            (math.sqrt(voxel.X**2 + voxel.Y**2) > fiduc_rad) |
                            (abs(voxel.Z) < veto_width)) # Extra veto for central cathode
        else:
            def fiducial_checker(voxel : icVoxel):
                return not ((voxel.Z < fiduc_z_min) |
                            (voxel.Z > fiduc_z_max) |
                            (math.sqrt(voxel.X**2 + voxel.Y**2) > fiduc_rad))

        return fiducial_checker

    def get_sensor_types(self) -> Sequence :
        return [key for key in self.sensor_ids.keys()]

    def get_sensor_ids(self, sensor_name : str) -> Sequence :
        assert sensor_name in self.sensor_ids.keys(), \
            f"Detector '{self.name}' has not sensor type '{sensor_name}'"
        return self.sensor_ids[sensor_name]



### Defining the detectors

# DEMOpp in all their versions
DEMOpp = \
    Detector( name               = 'DEMOpp',
              active_z_min       =   0.0  * units.mm,
              active_z_max       = 309.55 * units.mm,
              active_rad         =  97.1  * units.mm,
              energy_sensors     = ['PMT'],
              tracking_sensors   = ['SiPM'],
              sensor_ids         = {'PMT'  : (2, 4),
                                    'SiPM' : (14*1000, (17+1)*1000)},
              tracking_charge_th = 5.
            )

NEW = \
    Detector( name               = 'NEW',
              active_z_min       =   0.0 * units.mm,
              active_z_max       = 532.0 * units.mm,
              active_rad         = 208.0 * units.mm,
              energy_sensors     = ['PMT'],
              tracking_sensors   = ['SiPM'],
              sensor_ids         = {'PMT'  : (0, 12),
                                    'SiPM' : (1000, (29+1)*1000)},
              tracking_charge_th = 5.
            )

NEXT100 = \
    Detector( name               = 'NEXT100',
              active_z_min       =    0.0  * units.mm,
              active_z_max       = 1204.95 * units.mm,
              active_rad         =  492.0  * units.mm,
              energy_sensors     = ['PmtR11410'],
              tracking_sensors   = ['SiPM'],
              sensor_ids         = {'PmtR11410' : (0, 60),
                                    'SiPM'      : (1000, (57+1)*1000)},
              tracking_sns_pde   = 0.4,
              tracking_mask_att  = 0.0,
              tracking_charge_th = 5.
            )

FLEX100 = \
    Detector( name               = 'FLEX100',
              active_z_min       =    0.0  * units.mm,
              active_z_max       = 1204.95 * units.mm,
              active_rad         =  492.0  * units.mm,
              energy_sensors     = ['PmtR11410'],
              tracking_sensors   = ['TP_SiPM'],
              sensor_ids         = {'PmtR11410' : (0, 60),
                                    'TP_SiPM'   : (1000, 50000)},
              tracking_sns_pde   = 0.4,
              tracking_mask_att  = 0.0,
              tracking_charge_th = 5.
            )

FLEX100F = \
    Detector( name               = 'FLEX100F',
              active_z_min       =    0.0  * units.mm,
              active_z_max       = 1204.95 * units.mm,
              active_rad         =  492.0  * units.mm,
              energy_sensors     = ['F_SENSOR_L', 'F_SENSOR_R'],
              tracking_sensors   = ['TP_SiPM'],
              sensor_ids         = {'TP_SiPM'    : (  1000,  50000),
                                    'F_SENSOR_L' : (100000, 150000),
                                    'F_SENSOR_R' : (200000, 250000)},
              tracking_sns_pde   = 0.4,
              tracking_mask_att  = 0.0,
              tracking_charge_th = 5.
            )

NEXT500 = \
    Detector( name             = 'NEXT500',
              active_z_min     = -1000.0 * units.mm,
              active_z_max     =  1000.0 * units.mm,
              active_rad       =  1000.0 * units.mm,
              energy_sensors   = [],
              tracking_sensors = [],
              sensor_ids       = {}
            )

NEXT_2x2 = \
    Detector( name             = 'NEXT_2x2',
              active_z_min     = -1000.0 * units.mm,
              active_z_max     =  1000.0 * units.mm,
              active_rad       =  1000.0 * units.mm,
              energy_sensors   = [],
              tracking_sensors = [],
              sensor_ids       = {}
            )

NEXT_3X3 = \
    Detector( name             = 'NEXT_3X3',
              active_z_min     = -1500.0 * units.mm,
              active_z_max     =  1500.0 * units.mm,
              active_rad       =  1500.0 * units.mm,
              energy_sensors   = [],
              tracking_sensors = [],
              sensor_ids       = {}
            )

NEXT_HD = \
    Detector( name             = 'NEXT_HD',
              active_z_min     = -1300.0 * units.mm,
              active_z_max     =  1300.0 * units.mm,
              active_rad       =  1300.0 * units.mm,
              energy_sensors   = [],
              tracking_sensors = [],
              sensor_ids       = {}
            )



VALID_DETECTORS = ['DEMOpp', 'NEW', 'NEXT100', 'FLEX100', 'FLEX100F',
                   'NEXT500', 'NEXT_2x2', 'NEXT_3X3', 'NEXT_HD']



def get_detector(name : str):
    if name not in VALID_DETECTORS:
        raise DetectorNameNotDefined

    elif name == 'DEMOpp'   : return DEMOpp
    elif name == 'NEW'      : return NEW
    elif name == 'NEXT100'  : return NEXT100
    elif name == 'FLEX100'  : return FLEX100
    elif name == 'FLEX100F' : return FLEX100F
    elif name == 'NEXT500'  : return NEXT500
    elif name == 'NEXT_2x2' : return NEXT_2x2
    elif name == 'NEXT_3X3' : return NEXT_3X3
    elif name == 'NEXT_HD'  : return NEXT_HD

