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
    event_id       : int    = -1
    track_id       : int    = -1
    energy         : float  = np.nan
    length         : float  = np.nan
    num_voxels     : int    = 0
    t_ext1_x       : float  = np.nan
    t_ext1_y       : float  = np.nan
    t_ext1_z       : float  = np.nan
    t_ext2_x       : float  = np.nan
    t_ext2_y       : float  = np.nan
    t_ext2_z       : float  = np.nan
    blob1_x        : float  = np.nan
    blob1_y        : float  = np.nan
    blob1_z        : float  = np.nan
    blob1_energy   : float  = np.nan
    blob1_num_hits : int    = 0
    blob2_x        : float  = np.nan
    blob2_y        : float  = np.nan
    blob2_z        : float  = np.nan
    blob2_energy   : float  = np.nan
    blob2_num_hits : int    = 0
    ovlp_energy    : float  = np.nan


    @classmethod
    def from_icTrack(cls,
                     event_id : int,
                     track_id : int,
                     ic_track : Graph):
        "Creates a Track fron an icTrack"
        return cls(event_id, track_id,                  # ids
                   sum(voxel.E for voxel in ic_track),  # energy
                   track_length(ic_track),              # length
                   len(ic_track.nodes()))               # num_voxels


    def __repr__(self):
        s =  f"* Evt Id: {self.event_id} , Track id: {self.track_id}\n"
        s += f"  Energy: {self.energy / units.keV:.3f} keV "
        s += f"  Length: {self.length / units.mm:.3f} mm "
        s += f"  Num voxels: {self.num_voxels}\n"
        s += f"  True Ext1: ({self.t_ext1_x:.1f}, {self.t_ext1_y:.1f}, {self.t_ext1_z:.1f})"
        s += f"  True Ext2: ({self.t_ext2_x:.1f}, {self.t_ext2_y:.1f}, {self.t_ext2_z:.1f})\n"
        s += f"  Blob1: ({self.blob1_x:.1f}, {self.blob1_y:.1f}, {self.blob1_z:.1f}) "
        s += f"  Energy: {self.blob1_energy / units.keV:.3f} keV "
        s += f"  Num hits: {self.blob1_num_hits}\n"
        s += f"  Blob2: ({self.blob2_x:.1f}, {self.blob2_y:.1f}, {self.blob2_z:.1f}) "
        s += f"  Energy: {self.blob2_energy / units.keV:.3f} keV "
        s += f"  Num hits: {self.blob2_num_hits}\n"
        s += f"  Blobs ovlp energy: {self.ovlp_energy / units.keV:.3f} keV "
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
