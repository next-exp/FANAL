# General importings
import os
import sys
import glob
import json
import numpy         as np
import tables        as tb
import pandas        as pd

from dataclasses import dataclass

# IC importings
import invisible_cities.core.system_of_units               as units

from   invisible_cities.reco.tbl_functions  import filters as tbl_filters

from   invisible_cities.io.mcinfo_io        import load_mchits_df
from   invisible_cities.io.mcinfo_io        import load_mcparticles_df

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
                 det_name           : str    = '',
                 event_type         : str    = '',
                 input_fname        : str    = '',
                 output_fname       : str    = '',

                 trans_diff         : float  = np.nan,
                 long_diff          : float  = np.nan,
                 fwhm               : float  = np.nan,
                 e_min              : float  = np.nan,
                 e_max              : float  = np.nan,

                 voxel_size_x       : float  = np.nan,
                 voxel_size_y       : float  = np.nan,
                 voxel_size_z       : float  = np.nan,
                 strict_voxel_size  : bool   = False,
                 voxel_Eth          : float  = np.nan,

                 veto_width         : float  = np.nan,
                 veto_Eth           : float  = np.nan,

                 track_Eth          : float  = np.nan,
                 max_num_tracks     : int    = -1,
                 blob_radius        : float  = np.nan,
                 blob_Eth           : float  = np.nan,

                 roi_Emin           : float  = np.nan,
                 roi_Emax           : float  = np.nan,

                 # ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
                 verbosity          : str    = 'WARNING'
                ) :

        self.det_name           = det_name
        self.event_type         = event_type

        self.input_fname        = input_fname
        self.output_fname       = output_fname

        self.trans_diff         = trans_diff
        self.long_diff          = long_diff
        self.fwhm               = fwhm
        self.e_min              = e_min
        self.e_max              = e_max

        self.voxel_size_x       = voxel_size_x
        self.voxel_size_y       = voxel_size_y
        self.voxel_size_z       = voxel_size_z
        self.strict_voxel_size  = strict_voxel_size
        self.voxel_Eth          = voxel_Eth

        self.veto_width         = veto_width
        self.veto_Eth           = veto_Eth

        self.track_Eth          = track_Eth
        self.max_num_tracks     = max_num_tracks
        self.blob_radius        = blob_radius
        self.blob_Eth           = blob_Eth

        self.roi_Emin           = roi_Emin
        self.roi_Emax           = roi_Emax

        self.verbosity          = verbosity

        self._post_init()


    def _post_init(self):
        # The logger
        self.logger = get_logger('Fanal', self.verbosity)

        # Input files
        self.input_fnames = sorted(glob.glob(self.input_fname))
        self.input_path   = os.path.dirname(self.input_fnames[0])

        # Output path
        output_path = os.path.dirname(self.output_fname)
        if not os.path.isdir(output_path):
            print(f"  Making PATH {output_path}")
            os.makedirs(output_path)

        # The detector        
        self.detector         = get_detector(self.det_name)
        self.fiducial_checker = self.detector.get_fiducial_checker(self.veto_width)

        # Analysis parameters
        analysis_dict = {key: self.__dict__[key] for key in \
                         AnalysisParams.__dataclass_fields__.keys()}
        self.analysis_params = AnalysisParams(**analysis_dict)
        
        # DATA TO COLECT
        self.events_data   = EventList()
        self.tracks_data   = TrackList()
        self.voxels_data   = VoxelList()
        self.event_counter = EventCounter()

        # Assertions
        assert self.e_max >= self.e_min,       \
            "energy_filter settings not valid: 'e_max' must be higher than 'e_min'"
        assert self.roi_Emax >= self.roi_Emin, \
            "roi_filter settings not valid: 'roi_Emax' must be higher than 'roi_Emin'"


    @classmethod
    def from_config_file(cls, config_fname : str):
        "Initialize Setup from a conig file (json format)"
        with open(config_fname) as config_file:
            fanal_params = json.load(config_file)
            fanal_params['trans_diff']   = fanal_params['trans_diff']   * (units.mm / units.cm**.5)
            fanal_params['long_diff']    = fanal_params['long_diff']    * (units.mm / units.cm**.5)
            fanal_params['fwhm']         = fanal_params['fwhm']         * units.perCent
            fanal_params['e_min']        = fanal_params['e_min']        * units.keV
            fanal_params['e_max']        = fanal_params['e_max']        * units.keV
            fanal_params['voxel_size_x'] = fanal_params['voxel_size_x'] * units.mm
            fanal_params['voxel_size_y'] = fanal_params['voxel_size_y'] * units.mm
            fanal_params['voxel_size_z'] = fanal_params['voxel_size_z'] * units.mm
            fanal_params['voxel_Eth']    = fanal_params['voxel_Eth']    * units.keV
            fanal_params['veto_width']   = fanal_params['veto_width']   * units.mm
            fanal_params['veto_Eth']     = fanal_params['veto_Eth']     * units.keV
            fanal_params['track_Eth']    = fanal_params['track_Eth']    * units.keV
            fanal_params['blob_radius']  = fanal_params['blob_radius']  * units.mm
            fanal_params['blob_Eth']     = fanal_params['blob_Eth']     * units.keV
            fanal_params['roi_Emin']     = fanal_params['roi_Emin']     * units.keV
            fanal_params['roi_Emax']     = fanal_params['roi_Emax']     * units.keV
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
        params_to_store = ['det_name', 'event_type', 'input_fname', 'output_fname'] + \
                           list(AnalysisParams.__dataclass_fields__.keys())
        param_values = []
        for key in params_to_store:
            param_values.append(str(self.__dict__[key]))
        return pd.DataFrame(index=params_to_store, data=param_values, columns=['value'])


    def store_config(self):
        # It is stored with all the fields like 'str' to allow pandas
        # to place them in the same column
        self.config_df().to_hdf(self.output_fname, 'FANAL' + '/config',
                                data_columns = True, format = 'table')


    def store_data(self):
        # Storing data
        self.events_data.store(self.output_fname, 'FANAL')
        self.tracks_data.store(self.output_fname, 'FANAL')
        self.voxels_data.store(self.output_fname, 'FANAL')
        self.event_counter.store(self.output_fname, 'FANAL')


    def events_df(self):
        return self.events_data.df()


    def tracks_df(self):
        return self.tracks_data.df()


    def voxels_df(self):
        return self.voxels_data.df()


    def results_df(self):
        return self.event_counter.df()


    def run_analysis(self):
        # Print the Setup
        print(self)

        ### Opening the output file and storing configration parameters
        with tb.open_file(self.output_fname, 'w', filters=tbl_filters('ZLIB4')) as output_file:
            output_file.create_group('/', 'FANAL')
        self.store_config()

        ### Looping through all the input files
        verbose_every    = 1
        for input_fname in self.input_fnames:

            # Updating simulated and stored event counters
            configuration_df = pd.read_hdf(input_fname, '/MC/configuration', mode='r')
            self.event_counter.simulated += \
                int(configuration_df[configuration_df.param_key == 'num_events'].param_value)
            self.event_counter.stored    += \
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
                self.event_counter.analyzed += 1
                self.logger.info(f"*** Analyzing event Id: {event_id} ...")

                # Analyze event
                event_data, event_tracks, event_voxels = \
                    analyze_event(self.detector, int(event_id), self.event_type,
                                  file_mcParts.loc[event_id, :],
                                  file_mcHits.loc[event_id, :],
                                  self.fiducial_checker,
                                  self.analysis_params)

                # Storing event data
                self.events_data.add(event_data)
                self.tracks_data.add(event_tracks)
                self.voxels_data.add(event_voxels)

                # Verbosing
                if (not(self.event_counter.analyzed % verbose_every)):
                    print(f'* Num analyzed events: {self.event_counter.analyzed}')
                if (self.event_counter.analyzed == (10 * verbose_every)): verbose_every *= 10

        print(f'\n* Total analyzed events: {self.event_counter.analyzed}')

        # Filling filtered event counters
        self.event_counter.fill_filter_counters(self.events_data)

        ### STORING ANALYSIS DATA
        print(f'\n* Storing results in the output file ...\n  {output_file}\n')
        self.store_data()

        ### Ending ...
        print('\n* Analysis done !!\n')
        print(self.event_counter)



### Make it executable
if __name__ == '__main__':
    try:
        config_fname = sys.argv[1]
    except IndexError:
        print("\nUsage: python nexus-production.py config_file\n")
        sys.exit()

    fanal_setup = Setup.from_config_file(config_fname)
    fanal_setup.run_analysis()
