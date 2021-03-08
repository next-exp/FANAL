# General importings
import pandas as pd

from   typing import Tuple
from   typing import Callable

# IC importings
import invisible_cities.core.system_of_units      as units

from invisible_cities.evm.event_model         import MCHit
from invisible_cities.reco.paolina_functions  import voxelize_hits
from invisible_cities.reco.paolina_functions  import make_track_graphs
from invisible_cities.reco.paolina_functions  import blob_energies_hits_and_centres

# FANAL importings
from fanal.utils.logger             import get_logger

from fanal.core.detectors           import Detector
from fanal.core.fanal_types         import AnalysisParams

from fanal.containers.tracks        import TrackList
from fanal.containers.tracks        import track_from_ICtrack
from fanal.containers.voxels        import VoxelList
from fanal.containers.voxels        import voxel_from_ICvoxel
from fanal.containers.events        import Event

from fanal.analysis.mc_analysis     import check_mc_data
from fanal.analysis.mc_analysis     import reconstruct_hits
from fanal.analysis.voxel_analysis  import check_event_fiduciality
from fanal.analysis.voxel_analysis  import clean_voxels

# The logger
logger = get_logger('Fanal')



# TODO: Pass the whole fanal_setup instead all the config params
def analyze_event(detector          : Detector,
                  event_id          : int,
                  event_type        : str,
                  params            : AnalysisParams,
                  fiducial_checker  : Callable,
                  event_mcParts     : pd.DataFrame,
                  event_mcHits      : pd.DataFrame
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
        check_mc_data(event_mcHits, params.veto_Eth, params.e_min, params.e_max)
    if not event_data.mc_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the mc_filter ###
    # Reconstruct hits
    active_mcHits = event_mcHits[event_mcHits.label == 'ACTIVE']
    recons_hits   = reconstruct_hits(detector, active_mcHits, event_data.mc_energy,
                                     params.fwhm, params.trans_diff, params.long_diff)

    # Event smeared energy
    event_data.sm_energy     = recons_hits.energy.sum()
    event_data.energy_filter = (params.e_min <= event_data.sm_energy <= params.e_max)
    logger.info(f"smE: {event_data.sm_energy/units.keV:.1f} keV   " + \
                f"ENERGY filter: {event_data.energy_filter}")
    if not event_data.energy_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the energy_filter ###
    # Creating the IChits from reconstructed hits
    ic_hits = recons_hits.apply(lambda hit: \
        MCHit((hit.x, hit.y, hit.z), hit.time, hit.energy, 'ACTIVE'), axis=1).tolist()

    # Voxelizing using the ic_hits ...
    ic_voxels = voxelize_hits(ic_hits,
                              [params.voxel_size_x, params.voxel_size_y, params.voxel_size_z],
                              params.strict_voxel_size)

    # Cleaning voxels with energy < voxel_Eth
    ic_voxels = clean_voxels(ic_voxels, params.voxel_Eth)

    event_data.num_voxels = len(ic_voxels)
    eff_voxel_size = ic_voxels[0].size
    event_data.voxel_size_x = eff_voxel_size[0]
    event_data.voxel_size_y = eff_voxel_size[1]
    event_data.voxel_size_z = eff_voxel_size[2]
    logger.info(f"Num Voxels: {event_data.num_voxels:3}  of size: {eff_voxel_size} mm")

    # Check fiduciality
    event_data.veto_energy, event_data.fiduc_filter = \
        check_event_fiduciality(fiducial_checker, ic_voxels, params.veto_Eth)
    logger.info(f"Veto_E: {event_data.veto_energy/units.keV:.1f} keV   " + \
                f"FIDUC filter: {event_data.fiduc_filter}")

    if not event_data.fiduc_filter:
        # Storing voxels without track-id info
        for voxel_id in range(len(ic_voxels)):
            voxels_data.add(voxel_from_ICvoxel(event_id, -1, voxel_id, ic_voxels[voxel_id]))
        logger.debug(voxels_data)
        return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the fiduc_filter ###
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

    # Processing tracks: Getting energies, sorting and filtering ...
    event_data.num_tracks = tracks_data.len()

    # TODO: extend the conditions of the Track Filter to consider
    # track energy and track length (and anything else ?)
    event_data.track_filter = ((event_data.num_tracks >  0) &
                               (event_data.num_tracks <= params.max_num_tracks))

    logger.info(f"Num tracks: {event_data.num_tracks:3}  ->" + \
                f"  TRACK filter: {event_data.track_filter}")

    if not event_data.track_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the track_filter ###
    the_track = tracks_data.tracks[0]

    event_data.track_length = the_track.length

    # Getting & storing the track blob data
    ext1_energy, ext2_energy, ext1_hits, ext2_hits, ext1_pos, ext2_pos = \
        blob_energies_hits_and_centres(ic_tracks[0], params.blob_radius)

    the_track.ext1_energy, the_track.ext1_num_hits = ext1_energy, len(ext1_hits)
    the_track.ext1_x, the_track.ext1_y, the_track.ext1_z = \
        ext1_pos[0], ext1_pos[1], ext1_pos[2]

    the_track.ext2_energy, the_track.ext2_num_hits = ext2_energy, len(ext2_hits)
    the_track.ext2_x, the_track.ext2_y, the_track.ext2_z = \
        ext2_pos[0], ext2_pos[1], ext2_pos[2]

    event_data.blob1_energy, event_data.blob2_energy = ext1_energy, ext2_energy

    logger.info(tracks_data)

    # Applying the blob filter
    # TODO: extend the blob filter to check overlapping blobs
    event_data.blob_filter = (event_data.blob2_energy > params.blob_Eth)

    logger.info(f"Blob 1 energy: {event_data.blob1_energy/units.keV:4.1f} keV " + \
                f"  Blob 2 energy: {event_data.blob2_energy/units.keV:4.1f} keV"  + \
                f"  ->  BLOB filter: {event_data.blob_filter}")

    if not event_data.blob_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the blob_filter ###
    # Applying the ROI filter
    event_data.roi_filter = ((event_data.sm_energy >= params.roi_Emin) &
                                (event_data.sm_energy <= params.roi_Emax))

    logger.info(f"Event energy: {event_data.sm_energy/units.keV:6.1f} keV" + \
                f"  ->  ROI filter: {event_data.roi_filter}")

    return event_data, tracks_data, voxels_data
