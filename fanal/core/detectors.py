# General importings
import math

from dataclasses import dataclass

from typing      import Callable
from typing      import Sequence

# IC importings
import invisible_cities.core.system_of_units               as units
from   invisible_cities.evm.event_model       import Voxel as icVoxel

# FANAL importings
from fanal.core.fanal_exceptions import DetectorNameNotDefined


# Common constants to all detectors
# TODO: Make them detector dependant ??
S1_Eth         = 20. * units.keV  # Energy threshold for S1s
S1_WIDTH       = 10. * units.ns   # S1 time width
EVT_WIDTH      =  5. * units.ms   # Recorded time per event
DRIFT_VELOCITY =  1. * units.mm / units.mus
MIN_TIME_SHIFT =  1. * units.mus


@dataclass(frozen=True)
class Detector:
    name         : str
    active_z_min : float
    active_z_max : float
    active_rad   : float
    sensors      : dict

    def __post_init__(self):
        #self.symmetric : bool = (-self.active_z_min == self.active_z_max)
        # Defined this way and not the other one to be compatible with frozen=True
        object.__setattr__(self, "symmetric", (-self.active_z_min == self.active_z_max))

    def __repr__(self):
        s  = f"* Detector name: {self.name}\n"
        s += f"  ACTIVE dimensions (mm): Zmin = {self.active_z_min / units.mm:.2f} "
        s += f"  Zmax = {self.active_z_max / units.mm:.2f} "
        s += f"  Rad = {self.active_rad / units.mm:.2f}\n"
        s += f"  Symmetric: {self.symmetric}\n"
        s += f"  Sensor types: {[key for key in self.sensors.keys()]}\n"
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
        return [key for key in self.sensors.keys()]

    def get_sensor_ids(self, sensor_name : str) -> Sequence :
        assert sensor_name in self.sensors.keys(), \
            f"Detector '{self.name}' has not sensor type '{sensor_name}'"
        return self.sensors[sensor_name]



### Defining the detectors

# DEMOpp in all their versions
DEMOpp = \
    Detector( name         = 'DEMOpp',
              active_z_min =   0.0  * units.mm,
              active_z_max = 309.55 * units.mm,
              active_rad   =  97.1  * units.mm,
              sensors      = {'PMT'  : (2, 4),
                              'SiPM' : (14*1000, (17+1)*1000)}
            )

NEW = \
    Detector( name         = 'NEW',
              active_z_min =   0.0 * units.mm,
              active_z_max = 532.0 * units.mm,
              active_rad   = 208.0 * units.mm,
              sensors      = {'PMT'  : (0, 12),
                              'SiPM' : (1000, (29+1)*1000)}
            )

NEXT100 = \
    Detector( name         = 'NEXT100',
              active_z_min =    0.0  * units.mm,
              active_z_max = 1204.95 * units.mm,
              active_rad   =  492.0  * units.mm,
              sensors      = {'PmtR11410' : (0, 60),
                              'SiPM'      : (1000, (57+1)*1000)}
            )

FLEX100 = \
    Detector( name         = 'FLEX100',
              active_z_min =    0.0  * units.mm,
              active_z_max = 1204.95 * units.mm,
              active_rad   =  492.0  * units.mm,
              sensors      = {'PmtR11410' : (0, 60),
                              'TP_SiPM'   : (1000, 50000)}
            )

FLEX100F = \
    Detector( name         = 'FLEX100F',
              active_z_min =    0.0  * units.mm,
              active_z_max = 1204.95 * units.mm,
              active_rad   =  492.0  * units.mm,
              sensors      = {'TP_SiPM'    : (  1000,  50000),
                              'F_SENSOR_L' : (100000, 150000),
                              'F_SENSOR_R' : (200000, 250000)}
            )

NEXT500 = \
    Detector( name         = 'NEXT500',
              active_z_min = -1000.0 * units.mm,
              active_z_max =  1000.0 * units.mm,
              active_rad   =  1000.0 * units.mm,
              sensors      =  {}
            )

NEXT_2x2 = \
    Detector( name         = 'NEXT_2x2',
              active_z_min = -1000.0 * units.mm,
              active_z_max =  1000.0 * units.mm,
              active_rad   =  1000.0 * units.mm,
              sensors      =  {}
            )

NEXT_3X3 = \
    Detector( name         = 'NEXT_3X3',
              active_z_min = -1500.0 * units.mm,
              active_z_max =  1500.0 * units.mm,
              active_rad   =  1500.0 * units.mm,
              sensors      =  {}
            )

NEXT_HD = \
    Detector( name         = 'NEXT_HD',
              active_z_min = -1300.0 * units.mm,
              active_z_max =  1300.0 * units.mm,
              active_rad   =  1300.0 * units.mm,
              sensors      =  {}
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

