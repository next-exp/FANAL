# General importings
import numpy  as np
import pandas as pd

from   typing import Dict
from   typing import Any
from   typing import List


# IC importings
import invisible_cities.core.system_of_units      as units

from invisible_cities.evm.event_model         import MCHit
from invisible_cities.reco.paolina_functions  import voxelize_hits
from invisible_cities.reco.paolina_functions  import make_track_graphs
from invisible_cities.reco.paolina_functions  import length as track_length
from invisible_cities.reco.paolina_functions  import blob_energies

# FANAL importings
from fanal.core.fanal_types       import VolumeDim
from fanal.core.logger            import get_logger

from fanal.utils.fanal_units      import Qbb

from fanal.analysis.energy        import get_mc_energy
from fanal.analysis.energy        import smear_evt_energy
from fanal.analysis.position      import check_event_fiduciality

from fanal.reco.reco_io_functions import extend_voxels_reco_dict



# The logger
logger = get_logger('Fanal')



def get_event_data() -> Dict[str, Any]:
    """
    It returns a dictionary with a key for each field to be stored per event
    """
    event_data : Dict[str, Any] = {
        'event_id'     : np.nan,
        'num_MCparts'  : np.nan,
        'num_MChits'   : np.nan,
        'mcE'          : np.nan,
        'smE'          : np.nan,
        'smE_filter'   : False,
        'num_voxels'   : np.nan,
        'voxel_sizeX'  : np.nan,
        'voxel_sizeY'  : np.nan,
        'voxel_sizeZ'  : np.nan,
        'fid_filter'   : False,
        'num_tracks'   : np.nan,
        'tracks_filter': False,
        'track_E'      : np.nan,
        'track_voxels' : np.nan,
        'track_length' : np.nan,
        'blob1_E'      : np.nan,
        'blob2_E'      : np.nan,
        'blobs_filter' : False,
        'roi_filter'   : False
    }

    return event_data



def get_events_dict() -> Dict[str, List[Any]]:
    """
    It returns the dictionary to store the data from all the events.
    """
    event_data  = get_event_data()

    events_dict : Dict[str, List[Any]] = {}
    for key in event_data.keys():
        events_dict[key] = []

    return events_dict



def store_events_data(file_name   : str,
                      group_name  : str,
                      events_dict : Dict[str, List[Any]]
                     ) -> None:
    """
    Translates the events dictionary to a dataFrame that is stored in
    file_name / group_name / events.
    """
    # Creating the df
    events_df = pd.DataFrame(events_dict)

    # Formatting DF
    events_df.set_index('event_id', inplace = True)
    events_df.sort_index()

    # Storing DF
    events_df.to_hdf(file_name, group_name + '/events',
                     format = 'table', data_columns = True)

    print('  Total Events in File: {}'.format(len(events_df)))



def analyze_event(det_name          : str,
                  ACTIVE_dimensions : VolumeDim,
                  event_id          : int,
                  event_type        : str,
                  sigma_Qbb         : float,
                  e_min             : float,
                  e_max             : float,
                  voxel_size        : np.ndarray,
                  voxel_Eth         : float,
                  veto_width        : float,
                  min_veto_e        : float,
                  track_Eth         : float,
                  max_num_tracks    : int,
                  blob_radius       : float,
                  blob_Eth          : float,
                  roi_Emin          : float,
                  roi_Emax          : float,
                  event_mcParts     : pd.DataFrame,
                  event_mcHits      : pd.DataFrame,
                  voxels_dict       : Dict
                 )                 -> Dict:

    # Data to be filled
    event_data = get_event_data()

    # Filtering hits
    active_mcHits = event_mcHits[event_mcHits.label == 'ACTIVE'].copy()

    event_data['event_id']    = event_id
    event_data['num_MCparts'] = len(event_mcParts)
    event_data['num_MChits']  = len(active_mcHits)

    #event_data['mcE'] = active_mcHits.energy.sum() / units.keV
    event_data['mcE'] = get_mc_energy(active_mcHits)

    # Smearing the event energy
    event_data['smE'] = smear_evt_energy(event_data['mcE'], sigma_Qbb, Qbb)

    # Applying the smE filter
    event_data['smE_filter'] = (e_min <= event_data['smE'] <= e_max)

    # Verbosing
    logger.info(f"  Num mcHits: {event_data['num_MChits']:3}   "       + \
                f"mcE: {event_data['mcE']/units.keV:.1f} "             + \
                f"keV   smE: {event_data['smE']/units.keV:.1f} keV   " + \
                f"smE_filter: {event_data['smE_filter']}")

    # For those events passing the smE filter:
    if event_data['smE_filter']:

        # Creating the IChits with the smeared energies and translated Z positions
        IChits = active_mcHits.apply(lambda hit: MCHit((hit.x, hit.y, hit.z),
                                             hit.time, hit.energy, 'ACTIVE'), axis=1).tolist()

        # Voxelizing using the IChits ...
        event_voxels = voxelize_hits(IChits, voxel_size, strict_voxel_size=True)
        event_data['num_voxels'] = len(event_voxels)
        eff_voxel_size = event_voxels[0].size
        event_data['voxel_sizeX'] = eff_voxel_size[0]
        event_data['voxel_sizeY'] = eff_voxel_size[1]
        event_data['voxel_sizeZ'] = eff_voxel_size[2]

        # Storing voxels info
        for voxel_id in range(len(event_voxels)):
            extend_voxels_reco_dict(voxels_dict, event_id, voxel_id,
                                    event_voxels[voxel_id], voxel_Eth)

        # NON checking fiduciality
        #event_data['voxels_minZ'], event_data['voxels_maxZ'], event_data['voxels_maxRad'], \
        #event_data['veto_energy'], event_data['fid_filter'] = \
        #check_event_fiduciality(det_name, veto_width, min_veto_e, event_voxels)
        event_data['fid_filter'] = True

        # Verbosing Voxels
        logger.info(f"  NumVoxels: {event_data['num_voxels']:3}   "             + \
                    f"fid_filter: {event_data['fid_filter']}")
        logger.debug(event_voxels)

        # Make tracks
        event_tracks = make_track_graphs(event_voxels)
        num_tracks = len(event_tracks)
        event_data['num_tracks'] = num_tracks

        # Applying the tracks filter consisting on: (0 < num tracks < max_num_tracks)
        event_data['tracks_filter'] = ((num_tracks >  0) &
                                       (num_tracks <= max_num_tracks))

        # Verbosing
        logger.info(f"  Num tracks: {event_data['num_tracks']:2}  -->" + \
                    f"  tracks_filter: {event_data['tracks_filter']}")

        # For those events passing the tracks filter:
        if event_data['tracks_filter']:
            the_track = event_tracks[0]

            # Storing the track info
            event_data['track_E']      = sum(voxel.E for voxel in the_track)
            event_data['track_length'] = track_length(event_tracks[0])
            event_data['track_voxels'] = len(event_tracks[0].nodes())

            # Verbosing the track info
            logger.info(f"  Track energy: {event_data['track_E']:5.1f}  " + \
                        f"  Track length: {event_data['track_length']:5.1f}  " + \
                        f"  Track voxels: {event_data['track_voxels']:3}")

            # Getting the blob energies of the track
            event_data['blob1_E'], event_data['blob2_E'] = \
                blob_energies(the_track, blob_radius)

            # Applying the blobs filter
            event_data['blobs_filter'] = (event_data['blob2_E'] > blob_Eth)

            # Verbosing
            logger.info(f"  Blob 1 energy: {event_data['blob1_E']/units.keV:4.1f} keV " + \
                        f"  Blob 2 energy: {event_data['blob2_E']/units.keV:4.1f} keV"  + \
                        f"  -->  Blobs filter: {event_data['blobs_filter']}")

            ### For those events passing the blobs filter:
            if event_data['blobs_filter']:

                # Applying the ROI filter
                event_data['roi_filter'] = ((event_data['smE'] >= roi_Emin) &
                                            (event_data['smE'] <= roi_Emax))

                # Verbosing
                logger.info(f"  Event energy: {event_data['smE']/units.keV:6.1f} keV" + \
                            f"  -->  ROI filter: {event_data['roi_filter']}")

    return event_data
