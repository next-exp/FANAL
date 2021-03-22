# General importings
import os
import sys
import glob
import json
import tables      as tb
import pandas      as pd

from   typing  import Tuple


# IC importings
from invisible_cities.reco.tbl_functions  import filters as tbl_filters

from invisible_cities.io.mcinfo_io        import load_mchits_df
from invisible_cities.io.mcinfo_io        import load_mcparticles_df

# FANAL importings
from fanal.utils.logger             import get_logger
from fanal.utils.mc_utils           import get_event_numbers_in_file

from fanal.core.detectors           import get_detector
from fanal.core.fanal_types         import AnalysisParams

from fanal.containers.tracks        import TrackList
from fanal.containers.voxels        import VoxelList
from fanal.containers.events        import EventList
from fanal.containers.events        import EventCounter

from fanal.analysis.event_analysis  import analyze_event



class Setup:

    def __init__(self,
                 det_name        : str,
                 event_type      : str,
                 input_fname     : str,
                 output_fname    : str,
                 analysis_params : AnalysisParams,
                 verbosity       : str = 'WARNING'
                ) :

        # The detector
        self.det_name = det_name
        self.detector = get_detector(self.det_name)

        # Event type
        self.event_type = event_type

        # Input files
        self.input_fname  = input_fname
        self.input_fnames = sorted(glob.glob(self.input_fname))
        self.input_path   = os.path.dirname(self.input_fnames[0])

        # Output files
        self.output_fname = output_fname
        output_path = os.path.dirname(self.output_fname)
        if not os.path.isdir(output_path):
            print(f"  Making PATH {output_path}")
            os.makedirs(output_path)

        # Analysis params
        self.analysis_params = analysis_params

        # The logger
        self.logger = get_logger('Fanal', verbosity)


    @classmethod
    def from_config_file(cls, config_fname : str):
        "Initialize Setup from a conig file (json format)"
        # Loading file content
        with open(config_fname) as config_file:
            fanal_params = json.load(config_file)
        # Building the AnalysisParams
        analysis_dict = {key: fanal_params.pop(key) for key in \
                         AnalysisParams.__dataclass_fields__.keys()}
        fanal_params['analysis_params'] = AnalysisParams(**analysis_dict)
        fanal_params['analysis_params'].set_units()
        return cls(**fanal_params)


    def __repr__(self):
        s  =  "*******************************************************************************\n"
        s += f"*** Detector:          {self.det_name}\n"
        s += f"*** Reconstructing:    {self.event_type} events\n"
        s += f"*** Input  files:      {self.input_fname}  ({len(self.input_fnames)} files)\n"
        s += f"*** Output file:       {self.output_fname}\n"
        s += str(self.analysis_params)
        s +=  "*******************************************************************************\n"
        return s


    __str__ = __repr__


    def config_df(self):
        # Config params except the analysis-related ones
        params_to_store = ['det_name', 'event_type', 'input_fname', 'output_fname']
        param_values = []
        for key in params_to_store:
            param_values.append(str(self.__dict__[key]))
        # Adding analysis parameters
        params_to_store += list(AnalysisParams.__dataclass_fields__.keys())
        param_values    += list(str(val) for val in self.analysis_params.__dict__.values())

        return pd.DataFrame(index=params_to_store, data=param_values, columns=['value'])


    def store_config(self):
        # It is stored with all the fields like 'str' to allow pandas
        # to place them in the same column
        self.config_df().to_hdf(self.output_fname, 'FANAL' + '/config',
                                data_columns = True, format = 'table')


    def run_analysis(self) -> Tuple[pd.DataFrame,      # Event Counter
                                    pd.DataFrame,      # Event Data
                                    pd.DataFrame,      # Track Data
                                    pd.DataFrame] :    # Voxel Data

        ### Print the Setup
        print(self)

        ### Data to collect
        all_events    = EventList()
        all_tracks    = TrackList()
        all_voxels    = VoxelList()
        event_counter = EventCounter()

        ### Obtaining the fiducial_checker
        fiducial_checker = \
            self.detector.get_fiducial_checker(self.analysis_params.veto_width)

        ### Opening the output file and storing configration parameters
        with tb.open_file(self.output_fname, 'w', filters=tbl_filters('ZLIB4')) as output_file:
            output_file.create_group('/', 'FANAL')
        self.store_config()

        ### Looping through all the input files
        verbose_every    = 1
        for input_fname in self.input_fnames:

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
                self.logger.info(f"*** Analyzing event Id: {event_id} ...")

                # Analyze event
                event_data, event_tracks, event_voxels = \
                    analyze_event(self.detector, int(event_id), self.event_type,
                                  self.analysis_params, fiducial_checker,
                                  file_mcParts.loc[event_id, :],
                                  file_mcHits.loc[event_id, :])

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
        print(f'\n* Storing results in the output file ...\n  {output_file}\n')
        all_events   .store(self.output_fname, 'FANAL')
        all_tracks   .store(self.output_fname, 'FANAL')
        all_voxels   .store(self.output_fname, 'FANAL')
        event_counter.store(self.output_fname, 'FANAL')

        ### Ending ...
        print('\n* Analysis done !!\n')
        print(event_counter)

        return (event_counter.df(),
                all_events   .df(),
                all_tracks   .df(),
                all_voxels   .df())



### Make it executable
if __name__ == '__main__':
    try:
        config_fname = sys.argv[1]
    except IndexError:
        print("\nUsage: python nexus-production.py config_file\n")
        sys.exit()

    fanal_setup = Setup.from_config_file(config_fname)
    fanal_setup.run_analysis()
