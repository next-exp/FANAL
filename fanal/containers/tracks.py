# General importings
from __future__  import annotations

import pandas        as pd
import numpy         as np

from dataclasses import dataclass
from dataclasses import field

from typing      import List, Union

from networkx    import Graph

# IC importings
import invisible_cities.core.system_of_units                  as units
from   invisible_cities.reco.paolina_functions  import length as track_length




@dataclass
class Track:
    event_id      : int    = -1
    track_id      : int    = -1
    energy        : float  = np.nan
    length        : float  = np.nan
    num_voxels    : int    = 0
    ext1_x        : float  = np.nan
    ext1_y        : float  = np.nan
    ext1_z        : float  = np.nan
    ext1_energy   : float  = np.nan
    ext1_num_hits : int    = 0
    ext2_x        : float  = np.nan
    ext2_y        : float  = np.nan
    ext2_z        : float  = np.nan
    ext2_energy   : float  = np.nan
    ext2_num_hits : int    = 0

    def __repr__(self):
        s =  f"* Evt Id: {self.event_id} , Track id: {self.track_id}\n"
        s += f"  Energy: {self.energy / units.keV:.3f} keV "
        s += f"  Length: {self.length / units.mm:.3f} mm "
        s += f"  Num voxels: {self.num_voxels}\n"
        s += f"  Extr1: ({self.ext1_x:.1f},{self.ext1_y:.1f},{self.ext1_z:.1f}) "
        s += f"  Energy: {self.ext1_energy / units.keV:.3f} keV "
        s += f"  Num hits: {self.ext1_num_hits}\n"
        s += f"  Extr2: ({self.ext2_x:.1f},{self.ext2_y:.1f},{self.ext2_z:.1f}) "
        s += f"  Energy: {self.ext2_energy / units.keV:.3f} keV "
        s += f"  Num hits: {self.ext2_num_hits}"
        return s

    __str__ = __repr__



@dataclass
class TrackList:
    tracks : List[Track] = field(default_factory=list)

    def len(self):
        return len(self.tracks)

    def add(self, new_tracks: Union[Track, List[Track], TrackList]):
        if   isinstance(new_tracks, Track)     : self.tracks += [new_tracks]
        elif isinstance(new_tracks, list)      : self.tracks += new_tracks
        elif isinstance(new_tracks, TrackList) : self.tracks += new_tracks.tracks
        else:
            raise TypeError("Trying to add non-Track objects to TrackList")

    def df(self) -> pd.DataFrame:
        if len(self.tracks) == 0:
            print("* WARNING: Generating empty Tracks DataFrame")
            tracks_df = pd.DataFrame(columns=Track.__dataclass_fields__.keys())
        else:
            tracks_df = pd.DataFrame([track.__dict__ for track in self.tracks])
        tracks_df.set_index(['event_id', 'track_id'], inplace = True)
        tracks_df.sort_index()
        return tracks_df

    def store(self,
              file_name  : str,
              group_name : str):
        self.df().to_hdf(file_name, group_name + '/tracks',
                         format = 'table', data_columns = True)

    def __repr__(self):
        s = f"Track List with {len(self.tracks)} tracks ...\n"
        for track in self.tracks:
            s += str(track)
        return s

    __str__ = __repr__



def track_from_ICtrack(event_id : int,
                       track_id : int,
                       ic_track : Graph) -> Track:
    return Track(event_id, track_id,                  # ids
                 sum(voxel.E for voxel in ic_track),  # energy
                 track_length(ic_track),              # length
                 len(ic_track.nodes()))               # voxels