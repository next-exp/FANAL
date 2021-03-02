# General importings
import pandas as pd

from   typing import Tuple
from   typing import Callable

# IC importings
import invisible_cities.core.system_of_units      as units

from invisible_cities.evm.event_model         import MCHit
from invisible_cities.reco.paolina_functions  import voxelize_hits
from invisible_cities.reco.paolina_functions  import make_track_graphs
from invisible_cities.reco.paolina_functions  import blob_energies

# FANAL importings
from fanal.utils.logger              import get_logger

from fanal.containers.tracks         import TrackList
from fanal.containers.tracks         import track_from_ICtrack
from fanal.containers.voxels         import VoxelList
from fanal.containers.voxels         import voxel_from_ICvoxel
from fanal.containers.events         import Event

from fanal.analysis.mc_analysis      import check_mc_data
from fanal.analysis.mc_analysis      import reconstruct_hits
from fanal.analysis.voxel_analysis   import check_event_fiduciality

# The logger
logger = get_logger('Fanal')



# TODO: Pass the whole fanal_setup instead all the config params
def analyze_event(event_id          : int,
                  event_type        : str,
                  event_mcParts     : pd.DataFrame,
                  event_mcHits      : pd.DataFrame,
                  fwhm              : float,
                  e_min             : float,
                  e_max             : float,
                  voxel_size_x      : float,
                  voxel_size_y      : float,
                  voxel_size_z      : float,
                  strict_voxel_size : bool,
                  voxel_Eth         : float,
                  fiducial_checker  : Callable,
                  veto_Eth          : float,
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

    # Storing basic MC data
    event_data.event_id    = event_id
    event_data.num_mcParts = len(event_mcParts)
    event_data.num_mcHits  = len(event_mcHits)

    logger.info(f"Num mcParticles: {event_data.num_mcParts:3}   " + \
                f"Num mcHits: {event_data.num_mcHits:3}   ")

    # Processing MC data
    # TODO: Replace veto_Eth by buffer_Eth in this call
    event_data.mc_energy, event_data.mc_filter = \
        check_mc_data(event_mcHits, veto_Eth, e_min, e_max)
    if not event_data.mc_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the mc_filter
    # Reconstruct hits
    active_mcHits = event_mcHits[event_mcHits.label == 'ACTIVE']
    recons_hits   = reconstruct_hits(active_mcHits, event_data.mc_energy, fwhm)

    # Event smeared energy
    event_data.sm_energy     = recons_hits.energy.sum()
    event_data.energy_filter = (e_min <= event_data.sm_energy <= e_max)
    logger.info(f"smE: {event_data.sm_energy/units.keV:.1f} keV   " + \
                f"ENERGY filter: {event_data.energy_filter}")
    if not event_data.energy_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the energy_filter
    # Creating the IChits from reconstructed hits
    ic_hits = recons_hits.apply(lambda hit: \
        MCHit((hit.x, hit.y, hit.z), hit.time, hit.energy, 'ACTIVE'), axis=1).tolist()

    # Voxelizing using the ic_hits ...
    ic_voxels = voxelize_hits(ic_hits, [voxel_size_x, voxel_size_y, voxel_size_z],
                              strict_voxel_size)
    event_data.num_voxels = len(ic_voxels)
    eff_voxel_size = ic_voxels[0].size
    event_data.voxel_size_x = eff_voxel_size[0]
    event_data.voxel_size_y = eff_voxel_size[1]
    event_data.voxel_size_z = eff_voxel_size[2]
    logger.info(f"  Num Voxels: {event_data.num_voxels:3}  of size: {eff_voxel_size} mm")

    # Check fiduciality
    event_data.veto_energy, event_data.fiduc_filter = \
        check_event_fiduciality(fiducial_checker, ic_voxels, veto_Eth)
    logger.info(f"  Veto_E: {event_data.veto_energy/units.keV:.1f} keV   " + \
                f"fiduc_filter: {event_data.fiduc_filter}")

    if not event_data.fiduc_filter:
        # Storing voxels without track-id info
        for voxel_id in range(len(ic_voxels)):
            voxels_data.add(voxel_from_ICvoxel(event_id, -1, voxel_id, ic_voxels[voxel_id]))
        logger.debug(voxels_data)
        return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the fiduc_filter
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

    logger.info(f"Num tracks: {event_data.num_tracks:3}  -->" + \
                f"  TRACK filter: {event_data.track_filter}")

    if not event_data.track_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the track_filter
    the_track = ic_tracks[0]

    event_data.track_length = tracks_data.tracks[0].length

    # Getting the blob energies of the track
    event_data.blob1_E, event_data.blob2_E = \
        blob_energies(the_track, blob_radius)

    # Applying the blob filter
    event_data.blob_filter = (event_data.blob2_E > blob_Eth)

    logger.info(f"Blob 1 energy: {event_data.blob1_E/units.keV:4.1f} keV " + \
                f"  Blob 2 energy: {event_data.blob2_E/units.keV:4.1f} keV"  + \
                f"  -->  BLOB filter: {event_data.blob_filter}")

    if not event_data.blob_filter: return event_data, tracks_data, voxels_data

    ### For those events passing the blobs filter:
    # Applying the ROI filter
    event_data.roi_filter = ((event_data.sm_energy >= roi_Emin) &
                                (event_data.sm_energy <= roi_Emax))

    logger.info(f"Event energy: {event_data.sm_energy/units.keV:6.1f} keV" + \
                f"  -->  ROI filter: {event_data.roi_filter}")

    return event_data, tracks_data, voxels_data
