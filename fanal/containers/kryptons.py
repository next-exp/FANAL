# General importings
from __future__  import annotations

import pandas        as pd
import numpy         as np

from dataclasses import dataclass
from dataclasses import field

from typing      import List, Union

# IC importings
import invisible_cities.core.system_of_units   as units

# FANAL importings
from fanal.utils.types      import XYZ



@dataclass
class Krypton:
    krypton_id  : int   = -1
    true_x      : float = np.nan
    true_y      : float = np.nan
    true_z      : float = np.nan
    true_energy : float = np.nan
    rec_x       : float = np.nan
    rec_y       : float = np.nan
    rec_z       : float = np.nan
    rec_energy  : float = np.nan
    s2_pes      : int   = np.nan
    corr_s2_pes : int   = np.nan
    s1_pes      : int   = np.nan
    corr_s1_pes : int   = np.nan

    def __repr__(self):
        s =  f"* Krypton id: {self.krypton_id}\n"
        s += f"  True ... Position: ({self.true_x:.1f}, {self.true_y:.1f}, {self.true_z:.1f})  "
        s += f"  Energy: {self.true_energy / units.keV:.3f} keV\n"
        s += f"  Rec. ... Position: ({self.rec_x:.1f}, {self.rec_y:.1f}, {self.rec_z:.1f})  "
        s += f"  Energy: {self.rec_energy / units.keV:.3f} keV\n"
        s += f"  S2 pes: {self.s2_pes}  ->  Corrected: {self.corr_s2_pes}\n"
        s += f"  S1 pes: {self.s1_pes}  ->  Corrected: {self.corr_s1_pes}\n"
        return s

    __str__ = __repr__

    @property
    def true_pos(self):
        return XYZ(self.true_x, self.true_y, self.true_z)

    @property
    def rec_pos(self):
        return XYZ(self.rec_x, self.rec_y, self.rec_z)

    def set_true_pos(self, pos : XYZ):
        self.true_x, self.true_y, self.true_z = pos.x, pos.y, pos.z

    def set_rec_pos(self, pos : XYZ):
        self.rec_x, self.rec_y, self.rec_z = pos.x, pos.y, pos.z



@dataclass
class KryptonList:
    kryptons : List[Krypton] = field(default_factory=list)

    def len(self):
        return len(self.kryptons)

    def add(self, new_kryptons: Union[Krypton, List[Krypton], KryptonList]):
        if   isinstance(new_kryptons, Krypton)     : self.kryptons += [new_kryptons]
        elif isinstance(new_kryptons, list)        : self.kryptons += new_kryptons
        elif isinstance(new_kryptons, KryptonList) : self.kryptons += new_kryptons.kryptons
        else:
            raise TypeError("Trying to add non-Krypton objects to KryptonList")

    def df(self) -> pd.DataFrame:
        if len(self.kryptons) == 0:
            print("* WARNING: Generating empty Krypton DataFrame")
            kryptons_df = pd.DataFrame(columns=Krypton.__dataclass_fields__.keys())
        else:
            kryptons_df = pd.DataFrame([krypton.__dict__ for krypton in self.kryptons])
        kryptons_df.set_index('krypton_id', inplace = True)
        kryptons_df.sort_index()
        return kryptons_df

    def store(self,
              file_name  : str,
              group_name : str):
        self.df().to_hdf(file_name, group_name + '/kryptons',
                         format = 'table', data_columns = True)

    def __repr__(self):
        s = f"Krypton List with {len(self.kryptons)} kryptons ...\n"
        for krypton in self.kryptons:
            s += str(krypton)
        return s

    __str__ = __repr__
