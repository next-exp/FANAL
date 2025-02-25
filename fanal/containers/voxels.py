# General importings
from __future__  import annotations

import pandas        as pd
import numpy         as np
import networkx      as nx

from dataclasses import dataclass
from dataclasses import field

from typing      import List, Union

# IC importings
import invisible_cities.core.system_of_units               as units
from   invisible_cities.evm.event_model      import Voxel  as icVoxel

# FANAL importings
from fanal.utils.types                       import XYZ



@dataclass
class Voxel:
    event_id : int   = -1
    track_id : int   = -1
    voxel_id : int   = -1
    x        : float = np.nan
    y        : float = np.nan
    z        : float = np.nan
    energy   : float = np.nan

    @classmethod
    def from_icVoxel(cls,
                     event_id : int,
                     track_id : int,
                     voxel_id : int,
                     ic_voxel : icVoxel):
        return cls(event_id, track_id, voxel_id,  # ids
                   ic_voxel.X,                    # x coordinate
                   ic_voxel.Y,                    # y coordinate
                   ic_voxel.Z,                    # z coordinate
                   ic_voxel.E)                    # energy

    @classmethod
    def from_node(cls,
                  event_id : int,
                  track_id : int,
                  voxel_id : int,
                  node     : nx.Node):
        return cls(event_id, track_id, voxel_id,  # ids
                   node[0],                       # x coordinate
                   node[1],                       # y coordinate
                   node[2],                       # z coordinate
                   node[3])                       # energy

    def __repr__(self):
        s =  f"* Evt id: {self.event_id} , Trk id: {self.track_id} , Voxel id: {self.voxel_id}\n"
        s += f"  Position: ({self.x:.1f}, {self.y:.1f}, {self.z:.1f})  "
        s += f"  Energy: {self.energy / units.keV:.3f} keV\n"
        return s

    __str__ = __repr__

    @property
    def position(self):
        return XYZ(self.x, self.y, self.z)

    # For icVoxel compatibility
    @property
    def E(self): return self.energy
    @property
    def X(self): return self.x
    @property
    def Y(self): return self.y
    @property
    def Z(self): return self.z



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



@dataclass
class VoxelDF:
    df : pd.DataFrame = pd.DataFrame(columns=Voxel.__dataclass_fields__.keys())

    def __post_init__(self):
        self.df.set_index(['event_id', 'track_id', 'voxel_id'], inplace = True)
        self.df.sort_index()

    def add(self, new_voxels: Union[Voxel, List[Voxel], VoxelDF]):
        if   isinstance(new_voxels, Voxel):
            new_voxels_df = pd.DataFrame([new_voxels.__dict__])
            new_voxels_df.set_index(['event_id', 'track_id', 'voxel_id'], inplace = True)
            self.df = self.df.append(new_voxels_df)
        elif isinstance(new_voxels, List):
            new_voxels_df = pd.DataFrame([voxel.__dict__ for voxel in new_voxels])
            new_voxels_df.set_index(['event_id', 'track_id', 'voxel_id'], inplace = True)
            self.df = self.df.append(new_voxels_df)
        elif isinstance(new_voxels, VoxelDF):
            self.df = self.df.append(new_voxels.df)
        else:
            raise TypeError("Trying to add non-Voxel objects to VoxelDF")

    def store(self,
              file_name  : str,
              group_name : str):
        self.df.to_hdf(file_name, group_name + '/voxels',
                       format = 'table', data_columns = True)

    def __repr__(self):
        s  = f"Voxel DataFrame with {len(self.df)} voxels ...\n"
        s += str(self.df)
        return s

    __str__ = __repr__

