# General importings
from __future__  import annotations

import pandas        as pd
import numpy         as np

from dataclasses import dataclass
from dataclasses import field

from typing      import List, Union

# IC importings
import invisible_cities.core.system_of_units               as units
from   invisible_cities.evm.event_model      import Voxel  as icVoxel


@dataclass
class Voxel:
    event_id : int   = -1
    track_id : int   = -1
    voxel_id : int   = -1
    x        : float = np.nan
    y        : float = np.nan
    z        : float = np.nan
    energy   : float = np.nan

    def __repr__(self):
        s =  f"* Evt id: {self.event_id} , Trk id: {self.track_id} , Voxel id: {self.voxel_id}\n"
        s += f"  Position: ({self.x:.1f}, {self.y:.1f}, {self.z:.1f})  "
        s += f"  Energy: {self.energy / units.keV:.3f} keV\n"
        return s

    __str__ = __repr__



@dataclass
class VoxelList:
    voxels : List[Voxel] = field(default_factory=list)

    def len(self):
        return len(self.voxels)

    def add(self, new_voxels: Union[Voxel, List[Voxel], VoxelList]):
        if   isinstance(new_voxels, Voxel)     : self.voxels += [new_voxels]
        elif isinstance(new_voxels, list)      : self.voxels += new_voxels
        elif isinstance(new_voxels, VoxelList) : self.voxels += new_voxels.voxels
        else:
            raise TypeError("Trying to add non-Voxel objects to VoxelList")

    def df(self) -> pd.DataFrame:
        if len(self.voxels) == 0:
            print("* WARNING: Generating empty Voxels DataFrame")
            voxels_df = pd.DataFrame(columns=Voxel.__dataclass_fields__.keys())
        else:
            voxels_df = pd.DataFrame([voxel.__dict__ for voxel in self.voxels])
        voxels_df.set_index(['event_id', 'track_id', 'voxel_id'], inplace = True)
        voxels_df.sort_index()
        return voxels_df

    def store(self,
              file_name  : str,
              group_name : str):
        self.df().to_hdf(file_name, group_name + '/voxels',
                         format = 'table', data_columns = True)

    def __repr__(self):
        s = f"Voxel List with {len(self.voxels)} voxels ...\n"
        for voxel in self.voxels:
            s += str(voxel)
        return s

    __str__ = __repr__



def voxel_from_ICvoxel(event_id : int,
                       track_id : int,
                       voxel_id : int,
                       ic_voxel : icVoxel) -> Voxel:
    return Voxel(event_id, track_id, voxel_id,  # ids
                 ic_voxel.X,                    # x coordinate
                 ic_voxel.Y,                    # y coordinate
                 ic_voxel.Z,                    # z coordinate
                 ic_voxel.E)                    # energy
