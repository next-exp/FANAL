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
from fanal.utils.types              import XYZ

from fanal.core.detectors           import Detector
from fanal.core.fanal_types         import AnalysisParams

from fanal.containers.events        import Event
from fanal.containers.tracks        import Track
from fanal.containers.tracks        import TrackList
from fanal.containers.voxels        import Voxel
from fanal.containers.voxels        import VoxelList

from fanal.analysis.mc_analysis     import check_mc_data
from fanal.analysis.mc_analysis     import reconstruct_hits
from fanal.analysis.mc_analysis     import get_true_extrema
from fanal.analysis.mc_analysis     import order_true_extrema
from fanal.analysis.voxel_analysis  import check_event_fiduciality
from fanal.analysis.voxel_analysis  import check_event_fiduciality_df
from fanal.analysis.voxel_analysis  import clean_voxels

from fanal.paolina2.reco_functions  import voxelize_hits2
from fanal.paolina2.reco_functions  import make_track_graphs2
from fanal.paolina2.reco_functions  import get_and_store_blobs


# The logger
logger = get_logger('Fanal')




#############################################################################################
#############################################################################################
def analyze_event(detector          : Detector,
                  event_id          : int,
                  event_type        : str,
                  params            : AnalysisParams,
                  fiducial_checker  : Callable,
                  event_mcParts     : pd.DataFrame,
                  event_mcHits      : pd.DataFrame
                 )                 -> Tuple[Event, TrackList, VoxelList] :

    if   params.procedure == "paolina_ic":
        return analyze_event_ic(detector, event_id, event_type, params,
                                fiducial_checker, event_mcParts, event_mcHits)
    elif params.procedure == "paolina_2":
        return analyze_event_2(detector, event_id, event_type, params,
                               fiducial_checker, event_mcParts, event_mcHits)



#############################################################################################
#############################################################################################
def analyze_event_ic(detector          : Detector,
                     event_id          : int,
                     event_type        : str,
                     params            : AnalysisParams,
                     fiducial_checker  : Callable,
                     event_mcParts     : pd.DataFrame,
                     event_mcHits      : pd.DataFrame
                 )                    -> Tuple[Event, TrackList, VoxelList] :
    """
    It assess the global acceptance factor after fiducial, topology and ROI cuts
    based on the paolina functions implemented into IC.
    """
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
    event_data.mc_energy, event_data.mc_filter = \
        check_mc_data(event_mcHits, params.buffer_Eth, params.e_min, params.e_max)
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

    # Check fiduciality
    event_data.veto_energy, event_data.fiduc_filter = \
        check_event_fiduciality(fiducial_checker, ic_voxels, params.veto_Eth)
    logger.info(f"Veto_E: {event_data.veto_energy/units.keV:.1f} keV   " + \
                f"FIDUC filter: {event_data.fiduc_filter}")

    if not event_data.fiduc_filter:
        # Storing voxels without track-id info
        for voxel_id in range(len(ic_voxels)):
            voxels_data.add(Voxel.from_icVoxel(event_id, -1, voxel_id, ic_voxels[voxel_id]))
        logger.debug(voxels_data)
        return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the fiduc_filter ###
    # Make tracks
    ic_tracks  = make_track_graphs(ic_voxels)

    # Storing tracks from ic_tracks
    for track_id in range(len(ic_tracks)):
        ic_track = ic_tracks[track_id]
        tracks_data.add(Track.from_icTrack(event_id, track_id, ic_track))

        # Storing voxels from ic_voxels
        ic_voxels = list(ic_track.nodes())
        for voxel_id in range(len(ic_voxels)):
            voxels_data.add(Voxel.from_icVoxel(event_id, track_id, voxel_id,
                                               ic_voxels[voxel_id]))

    logger.debug(voxels_data)

    event_data.num_tracks = tracks_data.len()

    event_data.track_filter = ((event_data.num_tracks >  0) &
                               (event_data.num_tracks <= params.max_num_tracks))

    logger.info(f"Num tracks: {event_data.num_tracks:3}  ->" + \
                f"  TRACK filter: {event_data.track_filter}")

    if not event_data.track_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the track_filter ###
    the_track = tracks_data.tracks[0]

    # Getting & Storing Blobs info
    blob1_energy, blob2_energy, blob1_hits, blob2_hits, blob1_pos, blob2_pos = \
        blob_energies_hits_and_centres(ic_tracks[0], params.blob_radius)
    blob1_pos, blob2_pos = XYZ.from_array(blob1_pos), XYZ.from_array(blob2_pos)

    the_track.blob1_energy, the_track.blob1_num_hits = blob1_energy, len(blob1_hits)
    the_track.blob1_x, the_track.blob1_y, the_track.blob1_z = \
        blob1_pos.x, blob1_pos.y, blob1_pos.z
    the_track.blob2_energy, the_track.blob2_num_hits = blob2_energy, len(blob2_hits)
    the_track.blob2_x, the_track.blob2_y, the_track.blob2_z = \
        blob2_pos.x, blob2_pos.y, blob2_pos.z

    the_track.ovlp_energy = \
        float(sum(hit.E for hit in set(blob1_hits).intersection(set(blob2_hits))))

    # Getting & Storing True extrema info
    ext1, ext2 = get_true_extrema(event_mcParts, event_type)
    ext1, ext2 = order_true_extrema(ext1, ext2, blob1_pos, blob2_pos)

    the_track.t_ext1_x, the_track.t_ext1_y, the_track.t_ext1_z = ext1.x, ext1.y, ext1.z
    the_track.t_ext2_x, the_track.t_ext2_y, the_track.t_ext2_z = ext2.x, ext2.y, ext2.z

    # Storing Track info in event data
    event_data.track_length = the_track.length
    event_data.blob1_energy, event_data.blob2_energy = blob1_energy, blob2_energy

    logger.info(tracks_data)

    # Applying the blob filter
    event_data.blob_filter = ((event_data.blob2_energy > params.blob_Eth) &
                              (the_track.ovlp_energy == 0.))

    logger.info(f"Blob 1 energy: {event_data.blob1_energy/units.keV:4.1f} keV " + \
                f"  Blob 2 energy: {event_data.blob2_energy/units.keV:4.1f} keV"  + \
                f"  Overlap: {the_track.ovlp_energy/units.keV:4.1f} keV"  + \
                f"  ->  BLOB filter: {event_data.blob_filter}")

    if not event_data.blob_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the blob_filter ###
    # Applying the ROI filter
    event_data.roi_filter = ((event_data.sm_energy >= params.roi_Emin) &
                                (event_data.sm_energy <= params.roi_Emax))

    logger.info(f"Event energy: {event_data.sm_energy/units.keV:6.1f} keV" + \
                f"  ->  ROI filter: {event_data.roi_filter}")

    return event_data, tracks_data, voxels_data



#############################################################################################
#############################################################################################
def analyze_event_2(detector          : Detector,
                    event_id          : int,
                    event_type        : str,
                    params            : AnalysisParams,
                    fiducial_checker  : Callable,
                    event_mcParts     : pd.DataFrame,
                    event_mcHits      : pd.DataFrame
                   )                 -> Tuple[Event, TrackList, VoxelList] :
    """
    It assess the global acceptance factor after fiducial, topology and ROI cuts
    based on the paolina2 functions implemented into FANAL.
    Main differences respect to paolina_ic are:
    * Voxel position from the hits contained, not the centre.
    * Blob positions are voxels at the track extrema.
    * No need of IC event data model
    """
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
    event_data.mc_energy, event_data.mc_filter = \
        check_mc_data(event_mcHits, params.buffer_Eth, params.e_min, params.e_max)
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
    # Add an analysis parameter to use barycenter or not
    voxel_size = XYZ(params.voxel_size_x, params.voxel_size_y, params.voxel_size_z)
    tmp_voxels, eff_voxel_size = voxelize_hits2(recons_hits, voxel_size,
                                                baryc=params.barycenter)

    # Cleaning voxels with energy < voxel_Eth
    # TODO voxels = clean_voxels_df(voxels, params.voxel_Eth)

    event_data.num_voxels   = len(tmp_voxels)
    event_data.voxel_size_x = eff_voxel_size.x
    event_data.voxel_size_y = eff_voxel_size.y
    event_data.voxel_size_z = eff_voxel_size.z
    logger.info(f"Num Voxels: {event_data.num_voxels:3}  of size: {voxel_size} mm")

    # Check fiduciality
    event_data.veto_energy, event_data.fiduc_filter = \
        check_event_fiduciality_df(fiducial_checker, tmp_voxels, params.veto_Eth)
    logger.info(f"Veto_E: {event_data.veto_energy/units.keV:.1f} keV   " + \
                f"FIDUC filter: {event_data.fiduc_filter}")

    if not event_data.fiduc_filter:
        # Storing voxels without track-id info
        for voxel_id, voxel in tmp_voxels.iterrows():
            voxels_data.add(Voxel(event_id, -1, voxel_id, voxel.x,
                                  voxel.y, voxel.z, voxel.energy))
        logger.debug(voxels_data)
        return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the fiduc_filter ###
    # Make tracks
    graphs  = make_track_graphs2(tmp_voxels, contiguity=params.contiguity)

    # Storing tracks from graphs
    for graph_id in range(len(graphs)):
        graph = graphs[graph_id]
        tracks_data.add(Track.from_graph(event_id, graph_id, graph))

        # Storing voxels from ic_voxels
        nodes = list(graph.nodes())
        for node_id in range(len(nodes)):
            voxels_data.add(Voxel.from_node(event_id, graph_id, node_id, nodes[node_id]))

    logger.debug(voxels_data)

    event_data.num_tracks = tracks_data.len()

    event_data.track_filter = ((event_data.num_tracks >  0) &
                               (event_data.num_tracks <= params.max_num_tracks))

    logger.info(f"Num tracks: {event_data.num_tracks:3}  ->" + \
                f"  TRACK filter: {event_data.track_filter}")

    if not event_data.track_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the track_filter ###
    the_track = tracks_data.tracks[0]

    # Getting & Storing Blobs info
    blob1_pos, blob2_pos = get_and_store_blobs(the_track, graphs[0], params.blob_radius)

    # Getting & Storing True extrema info
    t_ext1, t_ext2 = get_true_extrema(event_mcParts, event_type)
    t_ext1, t_ext2 = order_true_extrema(t_ext1, t_ext2, blob1_pos, blob2_pos)

    the_track.t_ext1_x, the_track.t_ext1_y, the_track.t_ext1_z = t_ext1.x, t_ext1.y, t_ext1.z
    the_track.t_ext2_x, the_track.t_ext2_y, the_track.t_ext2_z = t_ext2.x, t_ext2.y, t_ext2.z

    # Storing Track info in event data
    event_data.track_length = the_track.length
    event_data.blob1_energy = the_track.blob1_energy
    event_data.blob2_energy = the_track.blob2_energy

    logger.info(tracks_data)

    # Applying the blob filter
    event_data.blob_filter = ((event_data.blob2_energy > params.blob_Eth) &
                              (the_track.ovlp_energy == 0.))

    logger.info(f"Blob 1 energy: {event_data.blob1_energy/units.keV:4.1f} keV " + \
                f"  Blob 2 energy: {event_data.blob2_energy/units.keV:4.1f} keV"  + \
                f"  Overlap: {the_track.ovlp_energy/units.keV:4.1f} keV"  + \
                f"  ->  BLOB filter: {event_data.blob_filter}")

    if not event_data.blob_filter: return event_data, tracks_data, voxels_data

    ### Continue analysis of events passing the blob_filter ###
    # Applying the ROI filter
    event_data.roi_filter = ((event_data.sm_energy >= params.roi_Emin) &
                                (event_data.sm_energy <= params.roi_Emax))

    logger.info(f"Event energy: {event_data.sm_energy/units.keV:6.1f} keV" + \
                f"  ->  ROI filter: {event_data.roi_filter}")

    return event_data, tracks_data, voxels_data
