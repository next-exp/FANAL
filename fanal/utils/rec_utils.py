import pandas as pd

from fanal.containers.events import Event
from fanal.containers.tracks import Track


def print_rec_event(event_id : int,
                    fname    : str
                   ):
    print(f"\nPrinting event {event_id} reconstructed contents ...\n")

    # Getting Rec DataFrames
    events_df = pd.read_hdf(fname, "FANAL" + '/events')
    tracks_df = pd.read_hdf(fname, "FANAL" + '/tracks')
    voxels_df = pd.read_hdf(fname, "FANAL" + '/voxels')

    # Event data
    print(Event(event_id, **events_df.loc[event_id]))

    # Tracks data
    try:
        evt_tracks = tracks_df.loc[event_id]
        print("* Tracks:\n")
        for track_id, track in evt_tracks.iterrows():
            print(Track(event_id, track_id, **evt_tracks.iloc[track_id]))
    except KeyError:
        print("* NO Tracks:\n")

    # Voxels data
    try:
        evt_voxels = voxels_df.loc[event_id]
        print("\n* Voxels:")
        print(evt_voxels)
    except KeyError:
        print("* NO Voxels:\n")
