import pandas     as pd

from   typing import Dict
from   typing import List
from   typing import Sequence
from   typing import Any
from   typing import Tuple

from networkx import Graph
from operator import itemgetter

# IC importings
import invisible_cities.core.system_of_units      as units

from invisible_cities.evm.event_model         import Voxel  as icVoxel

from invisible_cities.reco.paolina_functions  import length as track_length

# FANAL importings
from fanal.core.logger            import get_logger


# The logger
logger = get_logger('Fanal')


def get_voxels_dict() -> Dict[str, List[Any]]:
    """
    It returns a dictionary with a key for each field to be stored per voxel
    during the fanalIC reconstruction step.
    """
    voxels_dict : Dict[str, List[Any]] = {
        'event_id' : [],
        'voxel_id' : [],
        'X'        : [],
        'Y'        : [],
        'Z'        : [],
        'E'        : [],
        'negli'    : [],
        'newE'     : [],
        'track_id' : []
    }

    return voxels_dict



def extend_voxels_dict(voxels_dict : Dict[str, List[Any]],
                       event_id    : int,
                       track_id    : int,
                       voxel_id    : int,
                       voxel       : icVoxel,
                       voxel_Eth   : float,
                       newE        : float
                       ) -> None:
    """
    It stores all the data related to a voxel into the voxels_dict.
    """
    voxels_dict['event_id'].extend([event_id])
    voxels_dict['track_id'].extend([track_id])
    voxels_dict['voxel_id'].extend([voxel_id])
    voxels_dict['X']       .extend([voxel.X])
    voxels_dict['Y']       .extend([voxel.Y])
    voxels_dict['Z']       .extend([voxel.Z])
    voxels_dict['E']       .extend([voxel.E])
    voxels_dict['negli']   .extend([voxel.E < voxel_Eth])
    voxels_dict['newE']    .extend([newE])


#def extend_voxels_ana_dict(voxels_dict     : Dict[str, List[Any]],
#                           event_id        : int,
#                           voxel_id        : List[int],
#                           voxels_newE     : List[float],
#                           voxels_track_id : List[int]
#                          ) -> None:
#    """
#    It stores all the data related with the analysis of voxels
#    into the voxels_dict.
#    """
#
#    # Checking all Lists have the same length
#    assert (len(voxel_id) == len(voxels_newE) == len(voxels_track_id)), \
#        "extend_voxels_ana_data. All the lists must have the same length. {} {} {}" \
#        .format(len(voxel_id), len(voxels_newE), len(voxels_track_id))
#
#    voxels_dict['event_id'].extend([event_id] * len(voxel_id))
#    voxels_dict['voxel_id'].extend(voxel_id)
#    voxels_dict['newE']    .extend(voxels_newE)
#    voxels_dict['track_id'].extend(voxels_track_id)



def store_voxels_dict(file_name   : str,
                      voxels_dict : Dict[str, List[Any]]
                     ) -> None:
    """
    Translates the voxels dictionary to a dataFrame that is stored in
    file_name / group_name / voxels.
    """
    # Creating the DF
    voxels_df = pd.DataFrame(voxels_dict)

    # Formatting DF
    voxels_df.set_index(['event_id', 'voxel_id'], inplace = True)
    voxels_df.sort_index()

    # Storing DF
    #voxels_df.to_hdf(file_name, group_name + '/voxels', format='table',
    #                 data_columns='event_id')
    voxels_df.to_hdf(file_name, '/FANAL' + '/voxels',
                     format = 'table', data_columns = True)

    print('  Total Voxels in File: {}   (Negligible: {})\n'
          .format(len(voxels_df), len(voxels_df[voxels_df.negli == True])))



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

