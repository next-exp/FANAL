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
    x_true      : float = np.nan
    y_true      : float = np.nan
    z_true      : float = np.nan
    energy_true : float = 0.
    x_rec       : float = np.nan
    y_rec       : float = np.nan
    z_rec       : float = np.nan
    energy_rec  : float = 0.
    s1_pes      : int   = 0
    s2_pes      : int   = 0
    s1_pes_corr : int   = 0
    s2_pes_corr : int   = 0
    q_tot       : float = 0.
    q_max       : float = 0.

    def __repr__(self):
        s =  f"* Krypton id: {self.krypton_id}\n"
        s += f"  True ... Position: ({self.x_true:.1f}, {self.y_true:.1f}, {self.z_true:.1f})  "
        s += f"  Energy: {self.energy_true / units.keV:.3f} keV\n"
        s += f"  Rec. ... Position: ({self.x_rec:.1f}, {self.y_rec:.1f}, {self.z_rec:.1f})  "
        s += f"  Energy: {self.energy_rec / units.keV:.3f} keV\n"
        s += f"  S1 pes: {self.s1_pes}  ->  Corrected: {self.s1_pes_corr}\n"
        s += f"  S2 pes: {self.s2_pes}  ->  Corrected: {self.s2_pes_corr}\n"
        return s

    __str__ = __repr__

    @property
    def true_pos(self):
        return XYZ(self.x_true, self.y_true, self.z_true)

    @property
    def rec_pos(self):
        return XYZ(self.x_rec, self.y_rec, self.z_rec)

    def set_true_pos(self, pos : XYZ):
        self.x_true, self.y_true, self.z_true = pos.x, pos.y, pos.z

    def set_rec_pos(self, pos : XYZ):
        self.x_rec, self.y_rec, self.z_rec = pos.x, pos.y, pos.z



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
