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
import invisible_cities.core.system_of_units  as units

from   invisible_cities.reco.tbl_functions  import filters as tbl_filters

from   invisible_cities.io.mcinfo_io        import get_event_numbers_in_file
from   invisible_cities.io.mcinfo_io        import load_mchits_df
from   invisible_cities.io.mcinfo_io        import load_mcparticles_df

# FANAL importings
from fanal.utils.logger              import get_logger

from fanal.core.detectors           import get_detector

from fanal.containers.tracks        import TrackList
from fanal.containers.voxels        import VoxelList
from fanal.containers.events        import EventList
from fanal.containers.events        import EventCounter

from fanal.analysis.event_analysis  import analyze_event


# TODO - Remove it and implement Setup as a simple class ??
# TODO - Change init and __post_init to allow fanal_setup = Setup()
@dataclass
class Setup:
    det_name           : str    = ''
    event_type         : str    = ''

    input_fname        : str    = ''
    output_fname       : str    = ''

    fwhm               : float  = np.nan
    e_min              : float  = np.nan
    e_max              : float  = np.nan

    voxel_size_x       : float  = np.nan
    voxel_size_y       : float  = np.nan
    voxel_size_z       : float  = np.nan
    strict_voxel_size  : bool   = False
    voxel_Eth          : float  = np.nan

    veto_width         : float  = np.nan
    veto_Eth           : float  = np.nan

    track_Eth          : float  = np.nan
    max_num_tracks     : int    = -1
    blob_radius        : float  = np.nan
    blob_Eth           : float  = np.nan

    roi_Emin           : float  = np.nan
    roi_Emax           : float  = np.nan

    # ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    verbosity          : str    = 'WARNING'


    def __post_init__(self):

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


    def __repr__(self):
        s  =  "*******************************************************************************\n"
        s += f"*** Detector:          {self.det_name}\n"
        s += f"*** Reconstructing:    {self.event_type} events\n"
        s += f"*** Input  files:      {self.input_fname}  ({len(self.input_fnames)} files)\n"
        s += f"*** Output file:       {self.output_fname}\n"
        s += f"*** Energy Resolution: {self.fwhm / units.perCent:.2f}% fwhm at Qbb\n"
        s += f"*** Voxel Size:        ({self.voxel_size_x / units.mm}, "
        s += f"{self.voxel_size_y / units.mm}, {self.voxel_size_z / units.mm}) mm  "
        s += f"-  strict: {self.strict_voxel_size}\n"
        s += f"*** Voxel energy th.:  {self.voxel_Eth / units.keV:.1f} keV\n"
        s += f"*** Track energy th.:  {self.track_Eth / units.keV:.1f} keV\n"
        s += f"*** Max num Tracks:    {self.max_num_tracks}\n"
        s += f"*** Blob radius:       {self.blob_radius:.1f} mm\n"
        s += f"*** Blob energy th.:   {self.blob_Eth/units.keV:4.1f} keV\n"
        s += f"*** ROI energy limits: ({self.roi_Emin/units.keV:4.1f}, "
        s += f"{self.roi_Emax/units.keV:4.1f}) keV\n"
        s +=  "*******************************************************************************\n"
        return s

    __str__ = __repr__


#    def load_config(self, config_fname : str):
#        with open(config_fname) as config_file:
#            fanal_params = json.load(config_file)
#            fanal_params['fwhm']         = fanal_params['fwhm']         * units.perCent
#            fanal_params['e_min']        = fanal_params['e_min']        * units.keV
#            fanal_params['e_max']        = fanal_params['e_max']        * units.keV
#            fanal_params['voxel_size_x'] = fanal_params['voxel_size_x'] * units.mm
#            fanal_params['voxel_size_y'] = fanal_params['voxel_size_y'] * units.mm
#            fanal_params['voxel_size_z'] = fanal_params['voxel_size_z'] * units.mm
#            fanal_params['voxel_Eth']    = fanal_params['voxel_Eth']    * units.keV
#            fanal_params['veto_width']   = fanal_params['veto_width']   * units.mm
#            fanal_params['veto_Eth']     = fanal_params['veto_Eth']     * units.keV
#            fanal_params['track_Eth']    = fanal_params['track_Eth']    * units.keV
#            fanal_params['blob_radius']  = fanal_params['blob_radius']  * units.mm
#            fanal_params['blob_Eth']     = fanal_params['blob_Eth']     * units.keV
#            fanal_params['roi_Emin']     = fanal_params['roi_Emin']     * units.keV
#            fanal_params['roi_Emax']     = fanal_params['roi_Emax']     * units.keV
#        self.__init__(**fanal_params)


    def config_df(self):
        params_to_store = ['det_name', 'event_type', 'input_fname', 'output_fname',
                           'fwhm', 'e_min', 'e_max', 'voxel_size_x', 'voxel_size_y',
                           'voxel_size_z', 'strict_voxel_size', 'voxel_Eth',
                           'veto_width', 'veto_Eth', 'track_Eth', 'max_num_tracks',
                           'blob_radius', 'blob_Eth', 'roi_Emin', 'roi_Emax']
        param_values = []
        for key in params_to_store:
            param_values.append(str(self.__dict__[key]))
        return pd.DataFrame(index=params_to_store, data=param_values, columns=['value'])


    def store_config(self, output_fname : str):
        # It is stored with all the fields like 'str' to allow pandas
        # to place them in the same column
        self.config_df().to_hdf(output_fname, 'FANAL' + '/config',
                                data_columns = True, format = 'table')


    def events_df(self):
        return self.events_data.df()


    def tracks_df(self):
        return self.tracks_data.df()


    def voxels_df(self):
        return self.voxels_data.df()



    def run_analysis(self):
        # Print the Setup
        print(self)

        ### Opening the output file and storing configration parameters
        with tb.open_file(self.output_fname, 'w', filters=tbl_filters('ZLIB4')) as output_file:
            output_file.create_group('/', 'FANAL')
        self.store_config(self.output_fname)

        ### Looping through all the input files
        verbose_every    = 1
        for input_fname in self.input_fnames:

            # Updating simulated and stored event counters
            configuration_df = pd.read_hdf(input_fname, '/MC/configuration', mode='r')
            self.event_counter.simulated += int(configuration_df[configuration_df.param_key=='num_events'].param_value)
            self.event_counter.stored    += int(configuration_df[configuration_df.param_key=='saved_events'].param_value)

            # Getting event numbers
            file_event_ids = get_event_numbers_in_file(input_fname)
            print(f'\n*** Processing {input_fname}  ({len(file_event_ids)} events) ...')

            # Getting mc hits & particles
            file_mcHits  = load_mchits_df(input_fname)
            file_mcParts = load_mcparticles_df(input_fname)

            # Looping through all the events in iFile
            for event_id in file_event_ids:

                # Updating counter of analyzed events
                self.event_counter.analyzed += 1
                self.logger.info(f"Analyzing event Id: {event_id} ...")

                # Analyze event
                event_data, event_tracks, event_voxels = \
                    analyze_event(int(event_id), self.event_type,
                                  file_mcParts.loc[event_id, :],
                                  file_mcHits.loc[event_id, :],
                                  self.fwhm, self.e_min, self.e_max,
                                  self.voxel_size_x, self.voxel_size_y, self.voxel_size_z,
                                  self.strict_voxel_size, self.voxel_Eth,
                                  self.fiducial_checker, self.veto_Eth,
                                  self.track_Eth, self.max_num_tracks,
                                  self.blob_radius, self.blob_Eth,
                                  self.roi_Emin, self.roi_Emax)

                # Storing event data
                self.events_data.add(event_data)
                self.tracks_data.add(event_tracks)
                self.voxels_data.add(event_voxels)

                # Verbosing
                if (not(self.event_counter.analyzed % verbose_every)):
                    print(f'* Num analyzed events: {self.event_counter.analyzed}')
                if (self.event_counter.analyzed == (10 * verbose_every)): verbose_every *= 10


        ### STORING ANALYSIS DATA
        print(f'\n* Total analyzed events: {self.event_counter.analyzed}')

        # Storing events and voxels dataframes
        print(f'\n* Storing data in the output file ...\n  {output_file}\n')
        self.events_data.store(self.output_fname, 'FANAL')
        self.tracks_data.store(self.output_fname, 'FANAL')
        self.voxels_data.store(self.output_fname, 'FANAL')

        # Storing event counters as attributes
        events_df = self.events_data.df()
        self.event_counter.mc_filter     = len(events_df[events_df.mc_filter])
        self.event_counter.energy_filter = len(events_df[events_df.energy_filter])
        self.event_counter.fiduc_filter  = len(events_df[events_df.fiduc_filter])
        self.event_counter.track_filter  = len(events_df[events_df.track_filter])
        self.event_counter.blob_filter   = len(events_df[events_df.blob_filter])
        self.event_counter.roi_filter    = len(events_df[events_df.roi_filter])
        self.event_counter.store(self.output_fname, 'FANAL')

        ### Ending ...
        print('\n* Analysis done !!\n')
        print(self.event_counter)



# Make it executable
if __name__ == '__main__':
    try:
        config_fname = sys.argv[1]
    except IndexError:
        print("\nUsage: python nexus-production.py config_file\n")
        sys.exit()

    with open(config_fname) as config_file:
        fanal_params = json.load(config_file)
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

    fanal_setup = Setup(**fanal_params)
    fanal_setup.run_analysis()