# General importings
import numpy  as np
import pandas as pd

from   typing import Tuple


# IC importings
import invisible_cities.core.system_of_units      as units

from invisible_cities.evm.event_model         import MCHit
from invisible_cities.reco.paolina_functions  import voxelize_hits
from invisible_cities.reco.paolina_functions  import make_track_graphs
from invisible_cities.reco.paolina_functions  import blob_energies

# FANAL importings
from fanal.utils.logger       import get_logger

from fanal.core.fanal_types   import VolumeDim
from fanal.core.fanal_types   import DetName
from fanal.core.fanal_units   import Qbb

from fanal.analysis.energy    import get_mc_energy
from fanal.analysis.energy    import smear_evt_energy
from fanal.analysis.position  import check_event_fiduciality
from fanal.analysis.position  import translate_hit_positions

from fanal.containers.tracks  import TrackList
from fanal.containers.tracks  import track_from_ICtrack

from fanal.containers.voxels  import VoxelList
from fanal.containers.voxels  import voxel_from_ICvoxel

from fanal.containers.events  import Event


# The logger
logger = get_logger('Fanal')



def analyze_event(detector          : DetName,
                  ACTIVE_dimensions : VolumeDim,
                  event_id          : int,
                  event_type        : str,
                  event_mcParts     : pd.DataFrame,
                  event_mcHits      : pd.DataFrame,
                  sigma_Qbb         : float,
                  e_min             : float,
                  e_max             : float,
                  voxel_size_x      : float,
                  voxel_size_y      : float,
                  voxel_size_z      : float,
                  strict_voxel_size : bool,
                  voxel_Eth         : float,
                  veto_width        : float,
                  min_veto_e        : float,
                  track_Eth         : float,
                  max_num_tracks    : int,
                  blob_radius       : float,
                  blob_Eth          : float,
                  roi_Emin          : float,
                  roi_Emax          : float
                 )                 -> Tuple[Event, TrackList, VoxelList] :

    # Data to be filled
    event_data  = Event()
    tracks_data = TrackList()
    voxels_data = VoxelList()

    # Filtering hits
    active_mcHits = event_mcHits[event_mcHits.label == 'ACTIVE'].copy()

    event_data.event_id    = event_id
    event_data.num_mcParts = len(event_mcParts)
    event_data.num_mcHits  = len(active_mcHits)

    # The event mc energy is the sum of the energy of all the hits except
    # for Bi214 events, in which the number of S1 in the event is considered
    if (event_type == 'Bi214'):
        event_data.mc_energy = get_mc_energy(active_mcHits)
    else:
        event_data.mc_energy = active_mcHits.energy.sum()

    # Smearing the event energy
    event_data.sm_energy = smear_evt_energy(event_data.mc_energy, sigma_Qbb, Qbb)

    # Applying the smE filter
    event_data.energy_filter = (e_min <= event_data.sm_energy <= e_max)

    # Verbosing
    logger.info(f"Num mcHits: {event_data.num_mcHits:3}   " + \
                f"mcE: {event_data.mc_energy/units.keV:.1f} keV   " + \
                f"smE: {event_data.sm_energy/units.keV:.1f} keV   " + \
                f"energy_filter: {event_data.energy_filter}")

    # For those events passing the smE filter:
    if event_data.energy_filter:

        # Smearing hit energies
        smearing_factor = event_data.sm_energy / event_data.mc_energy
        active_mcHits['smE'] = active_mcHits['energy'] * smearing_factor

        # Translating hit Z positions from delayed hits
        translate_hit_positions(detector, active_mcHits)
        active_mcHits = active_mcHits[(active_mcHits.shifted_z < ACTIVE_dimensions.z_max) &
                                      (active_mcHits.shifted_z > ACTIVE_dimensions.z_min)]

        # Creating the IChits with the smeared energies and translated Z positions
        ic_hits = active_mcHits.apply(lambda hit: \
            MCHit((hit.x, hit.y, hit.shifted_z), hit.time, hit.smE, 'ACTIVE'), axis=1).tolist()

        # Voxelizing using the ic_hits ...
        ic_voxels = voxelize_hits(ic_hits, [voxel_size_x, voxel_size_y, voxel_size_z],
                                  strict_voxel_size)
        event_data.num_voxels = len(ic_voxels)
        eff_voxel_size = ic_voxels[0].size
        event_data.voxel_size_x = eff_voxel_size[0]
        event_data.voxel_size_y = eff_voxel_size[1]
        event_data.voxel_size_z = eff_voxel_size[2]

        # Check fiduciality
        event_data.voxels_min_z, event_data.voxels_max_z, event_data.voxels_max_rad, \
        event_data.veto_energy, event_data.fiduc_filter = \
        check_event_fiduciality(detector, veto_width, min_veto_e, ic_voxels)

        # Verbosing
        #logger.info(f"  Num Voxels: {event_data['num_voxels']:3}   "            + \
        #            f"minZ: {event_data['voxels_minZ']:.1f} mm   "              + \
        #            f"maxZ: {event_data['voxels_maxZ']:.1f} mm   "              + \
        #            f"maxR: {event_data['voxels_maxRad']:.1f} mm   "            + \
        #            f"veto_E: {event_data['veto_energy']/units.keV:.1f} keV   " + \
        #            f"fiduc_filter: {event_data['fiduc_filter']}")

        ### For those events NOT passing the fiducial filter
        # Storing voxels without track-id info
        if not event_data.fiduc_filter:
            for voxel_id in range(len(ic_voxels)):
                voxels_data.add(voxel_from_ICvoxel(event_id, -1, voxel_id, ic_voxels[voxel_id]))
            logger.debug(voxels_data)

        ### For those events passing the fiducial filter:
        else:

            # Make tracks
            ic_tracks  = make_track_graphs(ic_voxels)

            # Storing tracks from ic_tracks
            for track_id in range(len(ic_tracks)):
                ic_track = ic_tracks[track_id]
                tracks_data.add(track_from_ICtrack(event_id, track_id, ic_track))

                # Storing voxels from ic_voxels
                ic_voxels = list(ic_track.nodes())
                for voxel_id in range(len(ic_voxels)):
                    voxels_data.add(voxel_from_ICvoxel(event_id, track_id, voxel_id,
                                                       ic_voxels[voxel_id]))

            logger.debug(voxels_data)
            logger.info(tracks_data)

            # Processing tracks: Getting energies, sorting and filtering ...
            event_data.num_tracks = tracks_data.len()

            event_data.track_filter = ((event_data.num_tracks >  0) &
                                       (event_data.num_tracks <= max_num_tracks))

            # Verbosing
            logger.info(f"Num tracks: {event_data.num_tracks:3}  -->" + \
                        f"  track_filter: {event_data.track_filter}")

            ### For those events passing the track filter:
            if event_data.track_filter:
                the_track = ic_tracks[0]

                event_data.track_length = tracks_data.tracks[0].length

                # Getting the blob energies of the track
                event_data.blob1_E, event_data.blob2_E = \
                    blob_energies(the_track, blob_radius)

                # Applying the blob filter
                event_data.blob_filter = (event_data.blob2_E > blob_Eth)

                # Verbosing
                logger.info(f"Blob 1 energy: {event_data.blob1_E/units.keV:4.1f} keV " + \
                            f"  Blob 2 energy: {event_data.blob2_E/units.keV:4.1f} keV"  + \
                            f"  -->  Blob filter: {event_data.blob_filter}")


                ### For those events passing the blobs filter:
                if event_data.blob_filter:

                    # Applying the ROI filter
                    event_data.roi_filter = ((event_data.sm_energy >= roi_Emin) &
                                                (event_data.sm_energy <= roi_Emax))

                    # Verbosing
                    logger.info(f"Event energy: {event_data.sm_energy/units.keV:6.1f} keV" + \
                                f"  -->  ROI filter: {event_data.roi_filter}")

    return event_data, tracks_data, voxels_data
