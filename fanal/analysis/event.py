# General importings
import numpy  as np
import pandas as pd
import tables as tb

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
from fanal.core.fanal_types       import DetName
from fanal.core.logger            import get_logger

from fanal.utils.fanal_units      import Qbb

from fanal.analysis.energy        import get_mc_energy
from fanal.analysis.energy        import smear_evt_energy
from fanal.analysis.position      import check_event_fiduciality
from fanal.analysis.position      import translate_hit_positions

from fanal.analysis.voxel         import process_tracks



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
        'voxels_minZ'  : np.nan,
        'voxels_maxZ'  : np.nan,
        'voxels_maxRad': np.nan,
        'veto_energy'  : np.nan,
        'fid_filter'   : False,
        'num_tracks'   : np.nan,
        'tracks_filter': False,
        'track0_E'     : np.nan,
        'track0_length': np.nan,
        'track0_voxels': np.nan,
        'track1_E'     : np.nan,
        'track1_length': np.nan,
        'track1_voxels': np.nan,
        'track2_E'     : np.nan,
        'track2_length': np.nan,
        'track2_voxels': np.nan,
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



def extend_events_dict(
    events_dict : Dict[str, List[Any]],
    event_data  : Dict[str, Any]) -> None:
    """
    It stores all the data related to an event into the events_dict.
    The values not passed in the function called are set to default values
    to fill all the dictionary fields per event.
    """

    assert type(event_data['event_id']) == int, "event_id is mandatory"

    for key in event_data.keys():
        events_dict[key].extend([event_data[key]])



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



def store_events_counters(oFile                : tb.file.File,
                          simulated_events     : int,
                          stored_events        : int,
                          smE_filter_events    : int,
                          fid_filter_events    : int,
                          tracks_filter_events : int,
                          blobs_filter_events  : int,
                          roi_filter_events    : int
                         ) -> None:
    """
    Stores the event counters as attributes of oFile / group_name
    """
    oFile.set_node_attr('/FANAL', 'simulated_events',     simulated_events)
    oFile.set_node_attr('/FANAL', 'stored_events',        stored_events)
    oFile.set_node_attr('/FANAL', 'smE_filter_events',    smE_filter_events)
    oFile.set_node_attr('/FANAL', 'fid_filter_events',    fid_filter_events)
    oFile.set_node_attr('/FANAL', 'tracks_filter_events', tracks_filter_events)
    oFile.set_node_attr('/FANAL', 'blobs_filter_events',  blobs_filter_events)
    oFile.set_node_attr('/FANAL', 'roi_filter_events',    roi_filter_events)



def analyze_event(detector          : DetName,
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

    # The event mc energy is the sum of the energy of all the hits except
    # for Bi214 events, in which the number of S1 in the event is considered
    if (event_type == 'Bi214'):
        event_data['mcE'] = get_mc_energy(active_mcHits)
    else:
        event_data['mcE'] = active_mcHits.energy.sum()

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

        # Smearing hit energies
        smearing_factor = event_data['smE'] / event_data['mcE']
        active_mcHits['smE'] = active_mcHits['energy'] * smearing_factor

        # Translating hit Z positions from delayed hits
        translate_hit_positions(detector, active_mcHits)
        active_mcHits = active_mcHits[(active_mcHits.shifted_z < ACTIVE_dimensions.z_max) &
                                      (active_mcHits.shifted_z > ACTIVE_dimensions.z_min)]

        # Creating the IChits with the smeared energies and translated Z positions
        ic_hits = active_mcHits.apply(lambda hit: MCHit((hit.x, hit.y, hit.shifted_z),
                                                       hit.time, hit.smE, 'ACTIVE'), axis=1).tolist()

        # Voxelizing using the ic_hits ...
        ic_voxels = voxelize_hits(ic_hits, voxel_size, strict_voxel_size=True)
        event_data['num_voxels'] = len(ic_voxels)
        eff_voxel_size = ic_voxels[0].size
        event_data['voxel_sizeX'] = eff_voxel_size[0]
        event_data['voxel_sizeY'] = eff_voxel_size[1]
        event_data['voxel_sizeZ'] = eff_voxel_size[2]

        # Storing voxels info
        #for voxel_id in range(len(ic_voxels)):
        #    extend_voxels_dict(voxels_dict, event_id, voxel_id,
        #                       ic_voxels[voxel_id], voxel_Eth)

        # Check fiduciality
        event_data['voxels_minZ'], event_data['voxels_maxZ'], event_data['voxels_maxRad'], \
        event_data['veto_energy'], event_data['fid_filter'] = \
        check_event_fiduciality(detector, veto_width, min_veto_e, ic_voxels)

        # Verbosing
        logger.info(f"  Num Voxels: {event_data['num_voxels']:3}   "            + \
                    f"minZ: {event_data['voxels_minZ']:.1f} mm   "              + \
                    f"maxZ: {event_data['voxels_maxZ']:.1f} mm   "              + \
                    f"maxR: {event_data['voxels_maxRad']:.1f} mm   "            + \
                    f"veto_E: {event_data['veto_energy']/units.keV:.1f} keV   " + \
                    f"fid_filter: {event_data['fid_filter']}")

        for voxel in ic_voxels:
            logger.debug(f"    Voxel pos: ({voxel.X/units.mm:5.1f}, "               + \
                         f"{voxel.Y/units.mm:5.1f}, {voxel.Z/units.mm:5.1f}) mm   " + \
                         f"E: {voxel.E/units.keV:5.1f} keV")

        ### For those events passing the fiducial filter:
        if event_data['fid_filter']:

            # Make tracks
            event_tracks = make_track_graphs(ic_voxels)
            num_ini_tracks = len(event_tracks)
            logger.info(f"  Num initial tracks: {num_ini_tracks:2}")

            # Appending to every voxel, the track it belongs to
            #event_voxels_tracks = get_voxel_track_relations(event_voxels, event_tracks)

            # Appending ana-info to this event voxels
            #extend_voxels_ana_dict(voxels_dict, event_number, event_voxels.index.tolist(),
            #                        event_voxels_newE, event_voxels_tracks)

            # Processing tracks: Getting energies, sorting and filtering ...
            event_sorted_tracks = process_tracks(event_tracks, track_Eth)
            event_data['num_tracks'] = len(event_sorted_tracks)

            # Storing 3 hottest tracks info
            if event_data['num_tracks'] >= 1:
                event_data['track0_E']      = event_sorted_tracks[0][0]
                event_data['track0_length'] = event_sorted_tracks[0][1]
                event_data['track0_voxels'] = len(event_sorted_tracks[0][2].nodes())
                logger.info(f"  Track 0 energy: {event_data['track0_E']:5.1f}  " + \
                            f"  Track 0 length: {event_data['track0_length']:5.1f}  " + \
                            f"  Track 0 voxels: {event_data['track0_voxels']:3}")

            if event_data['num_tracks'] >= 2:
                event_data['track1_E']      = event_sorted_tracks[1][0]
                event_data['track1_length'] = event_sorted_tracks[1][1]
                event_data['track1_voxels'] = len(event_sorted_tracks[1][2].nodes())
                logger.info(f"  Track 1 energy: {event_data['track1_E']:5.1f}  " + \
                            f"  Track 1 length: {event_data['track1_length']:5.1f}  " + \
                            f"  Track 1 voxels: {event_data['track1_voxels']:3}")

            if event_data['num_tracks'] >= 3:
                event_data['track2_E']      = event_sorted_tracks[2][0]
                event_data['track2_length'] = event_sorted_tracks[2][1]
                event_data['track2_voxels'] = len(event_sorted_tracks[2][2].nodes())
                logger.info(f"  Track 2 energy: {event_data['track2_E']:5.1f}  " + \
                            f"  Track 2 length: {event_data['track2_length']:5.1f}  " + \
                            f"  Track 2 voxels: {event_data['track2_voxels']:3}")

            # Applying the tracks filter consisting on:
            # 0 < num tracks < max_num_tracks
            # the track length must be longer than 2 times the blob_radius
            event_data['tracks_filter'] = ((event_data['num_tracks'] >  0) &
                                           (event_data['num_tracks'] <= max_num_tracks) &
                                           (event_data['track0_length'] >=  2. * blob_radius))

            # Verbosing
            logger.info(f"  Num tracks: {event_data['num_tracks']:2}  -->" + \
                        f"  tracks_filter: {event_data['tracks_filter']}")

            ### For those events passing the tracks filter:
            if event_data['tracks_filter']:
                the_track = event_tracks[0]

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
