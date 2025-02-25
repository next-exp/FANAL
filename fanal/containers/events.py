# General importings
from __future__  import annotations

from dataclasses import dataclass
from dataclasses import field

from typing      import List
from typing      import Union

import pandas        as pd
import numpy         as np

# IC importings
import invisible_cities.core.system_of_units  as units



@dataclass
class Event:
    event_id       : int    = -1
    num_mcParts    : int    = -1
    num_mcHits     : int    = -1
    mc_energy      : float  = np.nan
    mc_filter      : bool   = False
    sm_energy      : float  = np.nan
    energy_filter  : bool   = False
    num_voxels     : int    = -1
    voxel_size_x   : float  = np.nan
    voxel_size_y   : float  = np.nan
    voxel_size_z   : float  = np.nan
    veto_energy    : float  = np.nan
    fiduc_filter   : bool   = False
    num_tracks     : int    = -1
    track_length   : float  = np.nan
    track_filter   : bool   = False
    blob1_energy   : float  = np.nan
    blob2_energy   : float  = np.nan
    blob_filter    : bool   = False
    roi_filter     : bool   = False

    def __repr__(self):
        s =  f"* Event id: {self.event_id}\n"
        s += f"  mcParts: {self.num_mcParts}   mcHits: {self.num_mcHits} "
        s += f"  mcE: {self.mc_energy / units.keV:.3f} keV  ->  MC Filter: {self.mc_filter}\n"
        s += f"  smE: {self.sm_energy / units.keV:.3f} keV  ->  Energy Filter: {self.energy_filter}\n"
        s += f"  Num voxels: {self.num_voxels}  of size:"
        s += f"  ({self.voxel_size_x / units.mm:.1f}, {self.voxel_size_x / units.mm:.1f},"
        s += f" {self.voxel_size_x / units.mm:.1f}) mm\n"
        s += f"  vetoE: {self.veto_energy / units.keV:.3f}  ->  Fiduc. Filter: {self.fiduc_filter}\n"
        s += f"  Num tracks: {self.num_tracks}  ->  Track Length: {self.track_length / units.mm:.1f} mm "
        s += f"  ->  Track Filter: {self.track_filter}\n"
        s += f"  Blob1 E: {self.blob1_energy / units.keV:.3f} keV "
        s += f"  Blob2 E: {self.blob2_energy / units.keV:.3f} keV "
        s += f"  ->  Blob Filter: {self.blob_filter}\n"
        s += f"  ROI Filter: {self.roi_filter}\n"
        return s

    __str__ = __repr__



@dataclass
class EventList:
    events : List[Event] = field(default_factory=list)

    def len(self):
        return len(self.events)

    def add(self, new_events: Union[Event, List[Event], EventList]):
        if   isinstance(new_events, Event)     : self.events += [new_events]
        elif isinstance(new_events, list)      : self.events += new_events
        elif isinstance(new_events, EventList) : self.events += new_events.events
        else:
            raise TypeError("Trying to add non-Event objects to EventList")

    def df(self) -> pd.DataFrame:
        if len(self.events) == 0:
            print("* WARNING: Generating empty Events DataFrame")
            events_df = pd.DataFrame(columns=Event.__dataclass_fields__.keys())
        else:
            events_df = pd.DataFrame([event.__dict__ for event in self.events])
        events_df.set_index('event_id', inplace = True)
        events_df.sort_index()
        return events_df

    def store(self,
              file_name  : str,
              group_name : str):
        self.df().to_hdf(file_name, group_name + '/events',
                         format = 'table', data_columns = True)

    def __repr__(self):
        s = f"Event List with {len(self.events)} events ...\n"
        for event in self.events:
            s += str(event)
        return s

    __str__ = __repr__



@dataclass
class EventDF:
    df : pd.DataFrame = pd.DataFrame(columns=Event.__dataclass_fields__.keys())

    def __post_init__(self):
        self.df.set_index('event_id', inplace = True)
        self.df.sort_index()

    def add(self, new_events: Union[Event, List[Event], EventDF]):
        if   isinstance(new_events, Event):
            new_events_df = pd.DataFrame([new_events.__dict__])
            new_events_df.set_index('event_id', inplace = True)
            self.df = self.df.append(new_events_df)
        elif isinstance(new_events, List):
            new_events_df = pd.DataFrame([event.__dict__ for event in new_events])
            new_events_df.set_index('event_id', inplace = True)
            self.df = self.df.append(new_events_df)
        elif isinstance(new_events, EventDF):
            self.df = self.df.append(new_events.df)
        else:
            raise TypeError("Trying to add non-Event objects to EventDF")

    def store(self,
              file_name  : str,
              group_name : str):
        self.df.to_hdf(file_name, group_name + '/events',
                       format = 'table', data_columns = True)

    def __repr__(self):
        s  = f"Event DataFrame with {len(self.df)} events ...\n"
        s += str(self.df)
        return s

    __str__ = __repr__



@dataclass
class EventCounter:
    simulated     : int = 0
    stored        : int = 0
    analyzed      : int = 0
    mc_filter     : int = 0
    energy_filter : int = 0
    fiduc_filter  : int = 0
    track_filter  : int = 0
    blob_filter   : int = 0
    roi_filter    : int = 0

    def __repr__(self):
        if not self.simulated: return "* Event counters EMPTY"
        s =   "* Event counters ...\n"
        s += f"  Simulated    : {self.simulated:10}  ({self.simulated/self.simulated:.2e})\n"
        s += f"  Stored       : {self.stored:10}  ({self.stored/self.simulated:.2e})\n"
        s += f"  Analyzed     : {self.analyzed:10}  ({self.analyzed/self.simulated:.2e})\n"
        s += f"  MC     filter: {self.mc_filter:10}  ({self.mc_filter/self.simulated:.2e})\n"
        s += f"  Energy filter: {self.energy_filter:10}  ({self.energy_filter/self.simulated:.2e})\n"
        s += f"  Fiduc. filter: {self.fiduc_filter:10}  ({self.fiduc_filter/self.simulated:.2e})\n"
        s += f"  Track  filter: {self.track_filter:10}  ({self.track_filter/self.simulated:.2e})\n"
        s += f"  Blob   filter: {self.blob_filter:10}  ({self.blob_filter/self.simulated:.2e})\n"
        s += f"  ROI    filter: {self.roi_filter:10}  ({self.roi_filter/self.simulated:.2e})\n"
        return s

    __str__ = __repr__

    def fill_filter_counters(self, event_data : Union[EventList, EventDF]):
        if isinstance(event_data, EventList): event_df = event_data.df()
        else                                : event_df = event_data.df
        self.mc_filter     = len(event_df[event_df.mc_filter])
        self.energy_filter = len(event_df[event_df.energy_filter])
        self.fiduc_filter  = len(event_df[event_df.fiduc_filter])
        self.track_filter  = len(event_df[event_df.track_filter])
        self.blob_filter   = len(event_df[event_df.blob_filter])
        self.roi_filter    = len(event_df[event_df.roi_filter])

    def df(self) -> pd.DataFrame:
        counters_df = pd.DataFrame([self.__dict__]).T
        counters_df.rename(columns={0: 'events'}, inplace = True)
        return counters_df

    def store(self,
              file_name  : str,
              group_name : str):
        self.df().to_hdf(file_name, group_name + '/results',
                         format = 'table', data_columns = True)
