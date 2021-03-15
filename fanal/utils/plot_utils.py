import pandas     as pd

import matplotlib
from   mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot        as plt
matplotlib.use('TkAgg')

import invisible_cities.core.system_of_units    as units
from   invisible_cities.io.mcinfo_io        import load_mchits_df
from   invisible_cities.io.mcinfo_io        import load_mcparticles_df

from   fanal.analysis.mc_analysis           import get_true_extrema
from   fanal.utils.mc_utils                 import get_fname_with_event



def plot_mc_event(event_id   : int,
                  fnames     : str,
                  event_type : str
                 ) -> None:
    """
    Plots the MC information of the event_id.
    """

    # Getting the right file
    fname = get_fname_with_event(event_id, fnames)
    if fname == '':
        print(f"\nEvent id: {event_id} NOT FOUND in input mc files.")
        return
    else:
        print(f"\nEvent id: {event_id}  contained in {fname}\n")

    # Getting the mcParticles and mcHits of the right event
    mcParts  = load_mcparticles_df(fname).loc[event_id]
    mcHits = load_mchits_df(fname).loc[event_id]

    # Plotting hits
    fig = plt.figure(figsize = (12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"MC event {event_id}")
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    p = ax.scatter(mcHits.x, mcHits.y, mcHits.z, c=(mcHits.energy / units.keV))
    cb = fig.colorbar(p, ax=ax)

    # Plotting True extrema
    ext1, ext2 = get_true_extrema(mcParts, event_type)
    ax.scatter3D(ext1[0], ext1[1], ext1[2], marker="*", lw=2, s=100, color='black')
    ax.scatter3D(ext2[0], ext2[1], ext2[2], marker="*", lw=2, s=100, color='black')

    cb.set_label('Energy (keV)')
    plt.show()
    return



def plot_rec_event(event_id : int,
                   fname    : str
                  ):
    """
    Plots the Reconstructed information of the event_id.
    """

    # Getting voxels and tracks of the right event
    voxels_df = pd.read_hdf(fname, "FANAL" + '/voxels')
    tracks_df = pd.read_hdf(fname, "FANAL" + '/tracks')

    voxels    = voxels_df.loc[event_id]
    tracks    = tracks_df.loc[event_id]

    # Plotting voxels
    fig = plt.figure(figsize = (12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Rec event {event_id}")
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    p = ax.scatter(voxels.x, voxels.y, voxels.z, marker="s",
                   s=(1000, 1000, 1000), c=(voxels.energy / units.keV))
    cb = fig.colorbar(p, ax=ax)

    # Plotting reconstructed blobs
    if len(tracks) == 1:
        track = tracks.loc[0]
        ax.scatter3D(track.blob1_x, track.blob1_y, track.blob1_z,
                     marker="*", lw=2, s=100, color='black')
        ax.scatter3D(track.blob2_x, track.blob2_y, track.blob2_z,
                     marker="*", lw=2, s=100, color='black')
    else:
        print("Non plotting blobs as the event has more than one track")


    cb.set_label('Energy (keV)')
    plt.show()
    return
