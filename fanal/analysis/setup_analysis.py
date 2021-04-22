# General importings
import pandas      as pd
from   typing  import Tuple
from   typing  import List

# IC importings
import invisible_cities.core.system_of_units as units
from   invisible_cities.io.mcinfo_io     import load_mchits_df
from   invisible_cities.io.mcinfo_io     import load_mcparticles_df
from   invisible_cities.io.mcinfo_io     import load_mcsensor_response_df

# FANAL importings
from fanal.core.detectors        import Detector
from fanal.core.fanal_units      import Qbb
from fanal.core.fanal_units      import kr_energy
from fanal.core.fanal_types      import BBAnalysisParams
from fanal.core.fanal_types      import KrAnalysisParams
from fanal.core.correction_maps  import build_correction_map
from fanal.core.correction_maps  import correct_s2

from fanal.utils.logger          import get_logger
from fanal.utils.mc_utils        import get_event_numbers_in_file

from fanal.containers.events     import EventCounter
from fanal.containers.events     import EventList
from fanal.containers.tracks     import TrackList
from fanal.containers.voxels     import VoxelList
from fanal.containers.kryptons   import KryptonList

from fanal.analysis.bb_analysis  import analyze_bb_event
from fanal.analysis.kr_analysis  import analyze_kr_event



# The logger
logger = get_logger('Fanal')



def run_bb_analysis(detector     : Detector,
                    input_fnames : List[str],
                    output_fname : str,
                    params       : BBAnalysisParams
                   )            -> Tuple[pd.DataFrame,    # Event Counter
                                         pd.DataFrame,    # Event Data
                                         pd.DataFrame,    # Track Data
                                         pd.DataFrame] :  # Voxel Data:

    ### Data to collect
    all_events    = EventList()
    all_tracks    = TrackList()
    all_voxels    = VoxelList()
    event_counter = EventCounter()

    ### Obtaining the fiducial_checker
    fiducial_checker = detector.get_fiducial_checker(params.veto_width)

    ### Looping through all the input files
    verbose_every = 1
    for input_fname in input_fnames:

        # Updating simulated and stored event counters
        configuration_df = pd.read_hdf(input_fname, '/MC/configuration', mode='r')
        event_counter.simulated += \
            int(configuration_df[configuration_df.param_key == 'num_events'].param_value)
        event_counter.stored    += \
            int(configuration_df[configuration_df.param_key == 'saved_events'].param_value)

        # Getting event ids
        event_ids = get_event_numbers_in_file(input_fname)
        print(f'\n*** Processing {input_fname}  ({len(event_ids)} events) ...\n')

        # Getting mc hits & particles
        file_mcHits  = load_mchits_df(input_fname)
        file_mcParts = load_mcparticles_df(input_fname)

        # Looping through all the events in current input file
        for event_id in event_ids:

            # Updating counter of analyzed events
            event_counter.analyzed += 1
            logger.info(f"*** Analyzing event Id: {event_id} ...")

            # Analyze event
            event_data, event_tracks, event_voxels = \
                analyze_bb_event(detector, int(event_id),
                                 params, fiducial_checker,
                                 file_mcParts.loc[event_id, :],
                                 file_mcHits .loc[event_id, :])

            # Storing event data
            all_events.add(event_data)
            all_tracks.add(event_tracks)
            all_voxels.add(event_voxels)

            # Verbosing num analyzed events
            if (not(event_counter.analyzed % verbose_every)):
                print(f'* Num analyzed events: {event_counter.analyzed}')
            if (event_counter.analyzed == (10 * verbose_every)): verbose_every *= 10

    print(f'\n* Total analyzed events: {event_counter.analyzed}')

    # Filling filtered event counters
    event_counter.fill_filter_counters(all_events)

    ### Storing global analysis data
    print(f'\n* Storing results in output file ...\n  {output_fname}\n')
    all_events   .store(output_fname, 'FANAL')
    all_tracks   .store(output_fname, 'FANAL')
    all_voxels   .store(output_fname, 'FANAL')
    event_counter.store(output_fname, 'FANAL')

    ### Ending ...
    print('\n* BB analysis done !!\n')
    print(event_counter)

    return (event_counter.df(),
            all_events   .df(),
            all_tracks   .df(),
            all_voxels   .df())



def run_kr_analysis(detector     : Detector,
                    input_fnames : List[str],
                    output_fname : str,
                    params       : KrAnalysisParams
                   )            -> pd.DataFrame :   # Krypton DataFrame

    ### Data to collect
    all_kryptons = KryptonList()

    ### Looping through all the input files
    verbose_every = 1
    for input_fname in input_fnames:

        # Getting event ids
        event_ids = get_event_numbers_in_file(input_fname)
        print(f'\n*** Processing {input_fname}  ({len(event_ids)} events) ...\n')

        sns_response = load_mcsensor_response_df(input_fname)
        mcHits       = load_mchits_df(input_fname)

        # Looping through all the events in current input file
        for event_id in event_ids:
            logger.info(f"Analyzing Kr event {event_id}")
            krypton_data = analyze_kr_event(detector,
                                            params,
                                            event_id,
                                            sns_response.loc[event_id, :],
                                            mcHits      .loc[event_id, :])
            all_kryptons.add(krypton_data)

    all_kryptons_df = all_kryptons.df()

    ### Building the correction map, and aplying it
    fiduc_selection = ((all_kryptons_df.z_rec   >= detector.active_z_min + params.veto_width) &
                       (all_kryptons_df.z_rec   <= detector.active_z_max - params.veto_width) &
                       (all_kryptons_df.rad_rec <= detector.active_rad   - params.veto_width))
    fiduc_kryptons = all_kryptons_df[fiduc_selection].copy()

    corr_map = build_correction_map(fiduc_kryptons,
                                    params.correction_map_type,
                                    num_bins=100)
    pes_corr = correct_s2(fiduc_kryptons,
                          corr_map,
                          params.correction_map_type)

    pes_to_MeV = (kr_energy/units.MeV) / fiduc_kryptons.s2_pes_corr.mean()
    fiduc_kryptons.energy_rec = fiduc_kryptons.s2_pes_corr * pes_to_MeV

    ### Storing global analysis data
    print(f'\n* Storing results in output file ...\n  {output_fname}\n')
    #fiduc_kryptons.store(output_fname, 'FANAL')
    fiduc_kryptons.to_hdf(output_fname, 'FANAL' + '/kryptons',
                          format = 'table', data_columns = True)


    ### Ending ...
    print('\n* Kr analysis done !!\n')

    return fiduc_kryptons