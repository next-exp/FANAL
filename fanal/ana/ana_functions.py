import os
import logging
import numpy as np
import pandas as pd

from operator import itemgetter

# Specific IC stuff
from invisible_cities.core.system_of_units_c  import units

logger = logging.getLogger('FanalAna')



def get_new_energies(event_voxels):
    """
    As a first approach, we give the energy of negligible voxels to
    the closest non-negligible voxel.
    """
    # Identify voxels with low energy
    negl_voxels = event_voxels[event_voxels.negli == True]
    
    negl_neig_pairs = []
    neig_index = []
    for i in negl_voxels.index:
        negl_voxel = event_voxels.loc[i]

        # looking the closest neighbour of every negligible voxel
        min_dist = 1000
        closest_index = i
        for j in event_voxels.index:
            if ((i != j) & (j not in negl_voxels.index)):
                dist = np.sqrt((negl_voxel.X-event_voxels.loc[j].X)**2 +
                               (negl_voxel.Y-event_voxels.loc[j].Y)**2 +
                               (negl_voxel.Z-event_voxels.loc[j].Z)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_index = j
        negl_neig_pairs.append((i, closest_index))
        neig_index.append(closest_index)

        logger.debug('    Negl. Voxel Id: {0} with E: {1:4.1f} keV  -->  Voxel Id: {2} '
            .format(i, negl_voxel.E / units.keV, closest_index))
            
    # Generating the list of new energies
    new_energies = []
    for i in event_voxels.index:
        # if negligible voxel -> new energy = 0
        if i in negl_voxels.index:
            new_energies.append(0.)
        # if voxel is the closest neigh. of any voxel -> new energy = old_energy + negligibles
        elif i in neig_index:
            new_voxelE = event_voxels.loc[i].E
            extraE = sum(event_voxels.loc[pair[0]].E for pair in negl_neig_pairs if i == pair[1])
            new_energies.append(new_voxelE + extraE)
        # The rest of voxels maintain their energies
        else:
            new_energies.append(event_voxels.loc[i].E)

    return new_energies



def get_voxel_track_relations(event_voxels, event_tracks):
    """
    It returns a list with the track id each voxel belongs to
    """
    voxel_tracks = []
    for i in event_voxels.index:
        if not event_voxels.loc[i].negli:
            found = False
            voxel_pos = (event_voxels.loc[i].X, event_voxels.loc[i].Y, event_voxels.loc[i].Z)
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



def process_tracks(event_tracks, track_Eth):
    """
    * Computing event energies
    * Filtering tracks with energy below threshold
    * Sorting tracks by energy
    Returning a list of tuples: (track_energy, track) of sorted tracks with energy
    higher than threshold ...
    """

    event_tracks_withE = []
    
    for i in range(len(event_tracks)):

        # Computing track energy
        track_E = sum(voxel.E for voxel in event_tracks[i])

        # If track energy >= threshold, append (track_E, track) to list
        # If not, iscarding tracks with track_E < threshold
        if track_E >= track_Eth:
            event_tracks_withE.append((track_E, event_tracks[i]))
        else:
            logger.debug('    Track with energy: {:6.1f} keV  -->  Discarded'.format(track_E/units.keV))

    # Sorting tracks by their energies
    event_tracks_withE = sorted(event_tracks_withE, key=itemgetter(0), reverse=True)

    # VERBOSING
    logger.debug('  Sorted-Good Tracks ...')
    for i in range(len(event_tracks_withE)):
        logger.debug('    Track {}  energy: {:6.1f} keV'.format(i, event_tracks_withE[i][0]/units.keV))

    return event_tracks_withE

