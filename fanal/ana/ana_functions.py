import logging
import numpy as np
import pandas as pd

from networkx import Graph
from operator import itemgetter
from typing   import Dict, List, Sequence, Tuple

import invisible_cities.core.system_of_units as units

from invisible_cities.evm.event_model         import Voxel
from invisible_cities.reco.paolina_functions  import length as track_length
from invisible_cities.reco.paolina_functions  import make_track_graphs
from invisible_cities.reco.paolina_functions  import blob_energies

from fanal.core.logger                        import get_logger
from fanal.ana.ana_io_functions               import get_event_ana_data
from fanal.ana.ana_io_functions               import extend_voxels_ana_dict


logger = get_logger('FanalAna')



def analyze_event(event_number      : int,
                  event_df          : pd.DataFrame,
                  event_voxels      : pd.DataFrame,
                  voxels_dict       : Dict,
                  track_Eth         : float,
                  max_num_tracks    : int,
                  blob_radius       : float,
                  blob_Eth          : float,
                  roi_Emin          : float,
                  roi_Emax          : float
                 ) -> Dict :
    
    # Data to be filled
    event_data = get_event_ana_data()

    event_data['event_id'] = event_number

    num_event_voxels = len(event_voxels)
    num_event_voxels_negli = len(event_voxels[event_voxels.negli])
    voxel_dimensions = (event_df.voxel_sizeX,
                        event_df.voxel_sizeY,
                        event_df.voxel_sizeZ)

    logger.info(f"  Total Voxels: {num_event_voxels}   Negli. Voxels: {num_event_voxels_negli}" + \
                f"   Voxels Size: ({voxel_dimensions[0]:3.1f}, {voxel_dimensions[1]:3.1f}, "    + \
                f"{voxel_dimensions[2]:3.1f}) mm")
    
    # If there is any negligible Voxel, distribute its energy between its neighbours,
    # if not, all voxels maintain their previous energies
    if num_event_voxels_negli:
        event_voxels_newE = get_new_energies(event_voxels)
    else:
        event_voxels_newE = event_voxels.E.tolist()

        
    # Translate fanalIC voxels info to IC voxels to make tracks
    #ic_voxels = [Voxel(event_voxels.iloc[i].X, event_voxels.iloc[i].Y, event_voxels.iloc[i].Z,
    #                   event_voxels_newE[i], voxel_dimensions) for i in range(num_event_voxels)]
    ic_voxels = []
    for i, voxel in event_voxels.iterrows():
        ic_voxel = Voxel(voxel.X, voxel.Y, voxel.Z, event_voxels_newE[i], voxel_dimensions)
        ic_voxels.append(ic_voxel)            

    # Make tracks
    event_tracks = make_track_graphs(ic_voxels)
    num_ini_tracks = len(event_tracks)
    logger.info(f"  Num initial tracks: {num_ini_tracks:2}")

    # Appending to every voxel, the track it belongs to
    event_voxels_tracks = get_voxel_track_relations(event_voxels, event_tracks)

    # Appending ana-info to this event voxels
    extend_voxels_ana_dict(voxels_dict, event_number, event_voxels.index.tolist(),
                            event_voxels_newE, event_voxels_tracks)

    # Processing tracks: Getting energies, sorting and filtering ...
    event_sorted_tracks = process_tracks(event_tracks, track_Eth)         
    event_data['num_tracks'] = len(event_sorted_tracks)

    # Getting 3 hottest tracks info
    if event_data['num_tracks'] >= 1:
        event_data['track0_E']      = event_sorted_tracks[0][0]
        event_data['track0_length'] = event_sorted_tracks[0][1]
        event_data['track0_voxels'] = len(event_sorted_tracks[0][2].nodes())
    if event_data['num_tracks'] >= 2:
        event_data['track1_E']      = event_sorted_tracks[1][0]
        event_data['track1_length'] = event_sorted_tracks[1][1]
        event_data['track1_voxels'] = len(event_sorted_tracks[1][2].nodes())
    if event_data['num_tracks'] >= 3:
        event_data['track2_E']      = event_sorted_tracks[2][0]
        event_data['track2_length'] = event_sorted_tracks[2][1]
        event_data['track2_voxels'] = len(event_sorted_tracks[2][2].nodes())
            
    # Applying the tracks filter consisting on:
    # 0 < num tracks < max_num_tracks
    # the track length must be longer than 2 times the blob_radius
    event_data['tracks_filter'] = ((event_data['num_tracks'] >  0) &
                                   (event_data['num_tracks'] <= max_num_tracks) &
                                   (event_data['track0_length'] >=  2. * blob_radius))
        
    # Verbosing
    logger.info(f"  Num final tracks: {event_data['num_tracks']:2}  -->" + \
                f"  tracks_filter: {event_data['tracks_filter']}")

            
    ### For those events passing the tracks filter:
    if event_data['tracks_filter']:

        # Getting the blob energies of the track with highest energy
        event_data['blob1_E'], event_data['blob2_E'] = \
            blob_energies(event_sorted_tracks[0][2], blob_radius)
                
        # Applying the blobs filter
        event_data['blobs_filter'] = (event_data['blob2_E'] > blob_Eth)
               
        # Verbosing
        logger.info(f"  Blob 1 energy: {event_data['blob1_E']/units.keV:4.1f} keV " + \
                    f"  Blob 2 energy: {event_data['blob2_E']/units.keV:4.1f} keV"  + \
                    f"  -->  Blobs filter: {event_data['blobs_filter']}")
                        
        ### For those events passing the blobs filter:
        if event_data['blobs_filter']:

            # Getting the total event smeared energy
            #event_smE = file_events.loc[event_number].smE
            event_smE = event_df.smE
                    
            # Applying the ROI filter
            event_data['roi_filter'] = ((event_smE >= roi_Emin) & (event_smE <= roi_Emax))
                
            # Verbosing
            logger.info(f"  Event energy: {event_smE/units.keV:6.1f} keV" + \
                        f"  -->  ROI filter: {event_data['roi_filter']}")

    return event_data



def get_new_energies(event_voxels : pd.DataFrame) -> List[float]:
    """
    It redistributes the energy of negligible voxels.
    As a first approach, we give the energy of negligible voxels to
    the closest non-negligible voxel.

    Parameters
    ----------
    event_voxels : pd.DataFrame
      Containing the voxels of an event.

    Returns
    -------
    A List with the new energies of all the incoming voxels
    """

    # Identify voxels with low energy
    negl_voxels = event_voxels[event_voxels.negli == True]

    negl_neig_pairs = []
    neig_index = []

    for i, negl_voxel in negl_voxels.iterrows():
        # looking the closest neighbour of every negligible voxel
        min_dist = 1000
        closest_index = i
        for j, event_voxel in event_voxels.iterrows():
            if ((i != j) & (j not in negl_voxels.index)):
                dist = np.sqrt((negl_voxel.X-event_voxel.X)**2 +
                               (negl_voxel.Y-event_voxel.Y)**2 +
                               (negl_voxel.Z-event_voxel.Z)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_index = j

        negl_neig_pairs.append((i, closest_index))
        neig_index.append(closest_index)

        logger.debug('    Negl. Voxel Id: {0} with E: {1:4.1f} keV  -->  Voxel Id: {2} '
                     .format(i, negl_voxel.E / units.keV, closest_index))

    # Generating the list of new energies
    new_energies = []
    for i, event_voxel in event_voxels.iterrows():
        # if negligible voxel -> new energy = 0
        if i in negl_voxels.index:
            new_energies.append(0.)
        # if voxel is the closest neigh. of any voxel ->
        # new energy = old_energy + negligibles
        elif i in neig_index:
            new_voxelE = event_voxel.E
            extraE = sum(event_voxels.loc[pair[0]].E for pair in negl_neig_pairs \
                     if i == pair[1])
            new_energies.append(new_voxelE + extraE)
        # The rest of voxels maintain their energies
        else:
            new_energies.append(event_voxel.E)

    return new_energies



def get_voxel_track_relations(event_voxels : pd.DataFrame,
                              event_tracks : Sequence[Graph]
                             ) -> List[int]:
    """
    It makes the association between voxels, and the track they belong to.

    Parameters
    ----------
    event_voxels : pd.DataFrame
      Containing the voxels of an event.
    event_tracks : Sequence[Graph]
      Containing the tracks of an event.

    Returns
    -------
    A list with the track id each voxel belongs to.
    """
    voxel_tracks = []
    for i, event_voxel in event_voxels.iterrows():
        if not event_voxel.negli:
            found = False
            voxel_pos = (event_voxel.X, event_voxel.Y, event_voxel.Z)
            # Look into each track
            for j in range(len(event_tracks)):
                for node in event_tracks[j].nodes():
                    node_pos = (node.X, node.Y, node.Z)
                    # If voxel in this track, append trackId and break
                    if voxel_pos == node_pos:
                        found = True
                        voxel_tracks.append(j)
                        break
                if found: break

        else:
            voxel_tracks.append(np.nan)

    return voxel_tracks



def process_tracks(event_tracks : Sequence[Graph],
                   track_Eth    : float
                  ) -> List[Tuple[float, float, Graph]]:
    """
    It calculates tracks energies.
    It filters tracks with length smaller than 2 times the blob_radius.
    It filters tracks with energy lower than threshold.

    Parameters
    ----------
    event_tracks : Sequence[Graph]
      Containing the tracks of an event.
    track_Eth : float
      Track energy threshold.

    Returns
    -------
    A list of tuples containing (track_energy, track_length, track)
    for all the tracks with energy higher than threshold,
    ordered per track energy.
    """

    event_tracks_withE = []

    for i in range(len(event_tracks)):

        # Computing track energy
        track_E = sum(voxel.E for voxel in event_tracks[i])

        # If track energy >= threshold, append (track_E, track) to list
        # If not, iscarding tracks with track_E < threshold
        if track_E >= track_Eth:
            track_l = track_length(event_tracks[i])
            event_tracks_withE.append((track_E, track_l, event_tracks[i]))
        else:
            logger.debug('    Track with energy: {:6.1f} keV  -->  Discarded'
                         .format(track_E/units.keV))

    # Sorting tracks by their energies
    event_tracks_withE = sorted(event_tracks_withE, key=itemgetter(0), reverse=True)

    # VERBOSING
    logger.debug('  Sorted-Good Tracks ...')
    for i in range(len(event_tracks_withE)):
        logger.debug('    Track {}  energy: {:6.1f} keV   length: {:6.1f} mm'
                     .format(i, event_tracks_withE[i][0]/units.keV,
                             event_tracks_withE[i][1]/units.mm))

    return event_tracks_withE
