# General importings
import math
import pandas as pd
import numpy  as np

from dataclasses import dataclass

from typing      import Callable
from typing      import Sequence
from typing      import List

# IC importings
import invisible_cities.core.system_of_units               as units
from   invisible_cities.evm.event_model       import Voxel as icVoxel

# FANAL importings
from fanal.core.tracking_maps                 import maps_path
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
    tracking_map_fname : str  = ''

    def __post_init__(self):
        #self.symmetric : bool = (-self.active_z_min == self.active_z_max)
        # Defined this way and not the other one to be compatible with frozen=True
        object.__setattr__(self, "symmetric", (-self.active_z_min == self.active_z_max))

        # assert all energy and tracking sensors have defined ids
        assert sorted(self.energy_sensors + self.tracking_sensors) == \
               sorted(self.sensor_ids.keys()), "Mismatch on sensor definitions"

        # Loading the tracking map
        tracking_map = pd.DataFrame()
        if (self.tracking_map_fname == ''):
            print("WARNING: No tracking map defined !!")
        else:
            tracking_map = pd.read_csv(maps_path + self.tracking_map_fname)
            tracking_map.set_index('sensor_id', inplace = True)
        object.__setattr__(self, "tracking_map", tracking_map)

    def __repr__(self):
        s  = f"* Detector name: {self.name}\n"
        s += f"  ACTIVE dimensions (mm): Zmin = {self.active_z_min / units.mm:.2f} "
        s += f"  Zmax = {self.active_z_max / units.mm:.2f} "
        s += f"  Rad = {self.active_rad / units.mm:.2f}\n"
        s += f"  Symmetric: {self.symmetric}\n"
        s += f"  Energy   Sensors  : {self.energy_sensors}\n"
        s += f"  Tracking Sensors  : {self.tracking_sensors}\n"
        s += f"  Tracking map fname: {self.tracking_map_fname}\n"
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



### Defining the detector parameters
VALID_DETECTORS = ['DEMOpp', 'NEW', 'NEXT100', 'FLEX100', 'FLEX100F',
                   'NEXT500', 'NEXT_2x2', 'NEXT_3X3', 'NEXT_HD']

# DEMOpp in all their versions
DEMOpp = {
    'name'               : 'DEMOpp',
    'active_z_min'       :   0.0  * units.mm,
    'active_z_max'       : 309.55 * units.mm,
    'active_rad'         :  97.1  * units.mm,
    'energy_sensors'     : ['PMT'],
    'tracking_sensors'   : ['SiPM'],
    'sensor_ids'         : {'PMT'  : (2, 4),
                            'SiPM' : (14*1000, (17+1)*1000)},
    'tracking_map_fname' : ''
}

NEW = {
    'name'               : 'NEW',
    'active_z_min'       :   0.0 * units.mm,
    'active_z_max'       : 532.0 * units.mm,
    'active_rad'         : 208.0 * units.mm,
    'energy_sensors'     : ['PMT'],
    'tracking_sensors'   : ['SiPM'],
    'sensor_ids'         : {'PMT'  : (0, 12),
                          'SiPM' : (1000, (29+1)*1000)},
    'tracking_map_fname' : ''
}

NEXT100 = {
    'name'               : 'NEXT100',
    'active_z_min'       :    0.0  * units.mm,
    'active_z_max'       : 1204.95 * units.mm,
    'active_rad'         :  492.0  * units.mm,
    'energy_sensors'     : ['PmtR11410'],
    'tracking_sensors'   : ['SiPM'],
    'sensor_ids'         : {'PmtR11410' : (0, 60),
                          'SiPM'      : (1000, (57+1)*1000)},
    'tracking_map_fname' : ''
}

FLEX100 = {
    'name'               : 'FLEX100',
    'active_z_min'       :    0.0  * units.mm,
    'active_z_max'       : 1204.95 * units.mm,
    'active_rad'         :  492.0  * units.mm,
    'energy_sensors'     : ['PmtR11410'],
    'tracking_sensors'   : ['TP_SiPM'],
    'sensor_ids'         : {'PmtR11410' : (0, 60),
                          'TP_SiPM'   : (1000, 50000)},
    'tracking_map_fname' : ''
}

FLEX100F = {
    'name'               : 'FLEX100F',
    'active_z_min'       :    0.0  * units.mm,
    'active_z_max'       : 1204.95 * units.mm,
    'active_rad'         :  492.0  * units.mm,
    'energy_sensors'     : ['F_SENSOR_L', 'F_SENSOR_R'],
    'tracking_sensors'   : ['TP_SiPM'],
    'sensor_ids'         : {'TP_SiPM'    : (  1000,  50000),
                          'F_SENSOR_L' : (100000, 150000),
                          'F_SENSOR_R' : (200000, 250000)},
    'tracking_map_fname' : 'FLEX100F.tracking_map.csv'
}

NEXT500 = {
    'name'               : 'NEXT500',
    'active_z_min'       : -1000.0 * units.mm,
    'active_z_max'       :  1000.0 * units.mm,
    'active_rad'         :  1000.0 * units.mm,
    'energy_sensors'     : [],
    'tracking_sensors'   : [],
    'sensor_ids'         : {},
    'tracking_map_fname' : ''
}

NEXT_2x2 = {
    'name'               : 'NEXT_2x2',
    'active_z_min'       : -1000.0 * units.mm,
    'active_z_max'       :  1000.0 * units.mm,
    'active_rad'         :  1000.0 * units.mm,
    'energy_sensors'     : [],
    'tracking_sensors'   : [],
    'sensor_ids'         : {},
    'tracking_map_fname' : ''
}

NEXT_3X3 = {
    'name'               : 'NEXT_3X3',
    'active_z_min'       : -1500.0 * units.mm,
    'active_z_max'       :  1500.0 * units.mm,
    'active_rad'         :  1500.0 * units.mm,
    'energy_sensors'     : [],
    'tracking_sensors'   : [],
    'sensor_ids'         : {},
    'tracking_map_fname' : ''
}

NEXT_HD = {
    'name'               : 'NEXT_HD',
    'active_z_min'       : -1300.0 * units.mm,
    'active_z_max'       :  1300.0 * units.mm,
    'active_rad'         :  1300.0 * units.mm,
    'energy_sensors'     : [],
    'tracking_sensors'   : [],
    'sensor_ids'         : {},
    'tracking_map_fname' : ''
}


def get_detector(name : str):
    if name not in VALID_DETECTORS:
        raise DetectorNameNotDefined
    elif name == 'DEMOpp'   : return Detector(**DEMOpp)
    elif name == 'NEW'      : return Detector(**NEW)
    elif name == 'NEXT100'  : return Detector(**NEXT100)
    elif name == 'FLEX100'  : return Detector(**FLEX100)
    elif name == 'FLEX100F' : return Detector(**FLEX100F)
    elif name == 'NEXT500'  : return Detector(**NEXT500)
    elif name == 'NEXT_2x2' : return Detector(**NEXT_2x2)
    elif name == 'NEXT_3X3' : return Detector(**NEXT_3X3)
    elif name == 'NEXT_HD'  : return Detector(**NEXT_HD)
