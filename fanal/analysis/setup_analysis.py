# General importings
import pandas      as pd
from   typing  import Tuple
from   typing  import List

# IC importings
from invisible_cities.io.mcinfo_io        import load_mchits_df
from invisible_cities.io.mcinfo_io        import load_mcparticles_df

# FANAL importings
from fanal.core.detectors        import Detector
from fanal.core.fanal_types      import BBAnalysisParams
from fanal.core.fanal_types      import KrAnalysisParams

from fanal.utils.logger          import get_logger
from fanal.utils.mc_utils        import get_event_numbers_in_file

from fanal.containers.events     import EventCounter
from fanal.containers.events     import EventList
from fanal.containers.tracks     import TrackList
from fanal.containers.voxels     import VoxelList

from fanal.analysis.bb_analysis  import analyze_bb_event
from fanal.analysis.kr_analysis  import analyze_kr_event

# The logger
logger = get_logger('Fanal')



def run_bb_analysis(detector     : Detector,
                    event_type   : str,
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

        # Getting event numbers
        file_event_ids = get_event_numbers_in_file(input_fname)
        print(f'\n*** Processing {input_fname}  ({len(file_event_ids)} events) ...\n')

        # Getting mc hits & particles
        file_mcHits  = load_mchits_df(input_fname)
        file_mcParts = load_mcparticles_df(input_fname)

        # Looping through all the events in iFile
        for event_id in file_event_ids:

            # Updating counter of analyzed events
            event_counter.analyzed += 1
            logger.info(f"*** Analyzing event Id: {event_id} ...")

            # Analyze event
            event_data, event_tracks, event_voxels = \
                analyze_bb_event(detector, int(event_id), event_type,
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
