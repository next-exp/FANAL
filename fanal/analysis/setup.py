# General importings
import os
import sys
import glob
import json
import numpy         as np
import tables        as tb
import pandas        as pd
from   typing    import Tuple
from   typing    import List

from dataclasses import dataclass
from dataclasses import field


# IC importings
import invisible_cities.core.system_of_units  as units

from invisible_cities.reco.tbl_functions  import filters as tbl_filters

from invisible_cities.io.mcinfo_io        import get_event_numbers_in_file
from invisible_cities.io.mcinfo_io        import load_mchits_df
from invisible_cities.io.mcinfo_io        import load_mcparticles_df

# FANAL importings
from fanal.utils.fanal_units      import Qbb

from fanal.core.logger            import get_logger
from fanal.core.detector          import get_active_size
from fanal.core.detector          import get_fiducial_size
from fanal.core.fanal_types       import DetName

from fanal.analysis.event         import analyze_event
from fanal.analysis.tracks        import TrackList
from fanal.analysis.voxels        import VoxelList
from fanal.analysis.events        import EventList
from fanal.analysis.events        import EventCounter



@dataclass
class Setup:
    det_name           : str    = ''
    event_type         : str    = ''

    input_fname        : str    = ''
    output_fname       : str    = ''

    fwhm               : float  = np.nan
    e_min              : float  = np.nan
    e_max              : float  = np.nan

    voxel_size         : List[float] = field(default_factory=list)
    strict_voxel_size  : bool   = False
    voxel_Eth          : float  = np.nan

    veto_width         : float  = np.nan
    min_veto_e         : float  = np.nan

    track_Eth          : float  = np.nan
    max_num_tracks     : int    = -1
    blob_radius        : float  = np.nan
    blob_Eth           : float  = np.nan

    roi_Emin           : float  = np.nan
    roi_Emax           : float  = np.nan

    verbosity          : str    = 'WARNING' # ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')


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
        self.detector          = getattr(DetName, self.det_name)
        self.active_dimensions = get_active_size(self.detector)
        self.fid_dimensions    = get_fiducial_size(self.detector, self.veto_width)

        # Reconstruction
        self.sigma_Qbb = self.fwhm * Qbb / 2.355

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
        s += f"*** Energy Resolution: {self.fwhm / units.perCent:.2f}% fwhm at Qbb  ->  "
        s += f"Sigma: {self.sigma_Qbb/units.keV:.3f} keV\n"
        s += f"*** Voxel Size:        ({self.voxel_size[0] / units.mm}, "
        s += f"{self.voxel_size[1] / units.mm}, {self.voxel_size[2] / units.mm}) mm  "
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
#            fanal_params['fwhm']        = fanal_params['fwhm']        * units.perCent
#            fanal_params['e_min']       = fanal_params['e_min']       * units.keV
#            fanal_params['e_max']       = fanal_params['e_max']       * units.keV
#            fanal_params['voxel_size']  = np.array(fanal_params['voxel_size'])  * units.mm
#            fanal_params['voxel_Eth']   = fanal_params['voxel_Eth']   * units.keV
#            fanal_params['veto_width']  = fanal_params['veto_width']  * units.mm
#            fanal_params['min_veto_e']  = fanal_params['min_veto_e']  * units.keV
#            fanal_params['track_Eth']   = fanal_params['track_Eth']   * units.keV
#            fanal_params['blob_radius'] = fanal_params['blob_radius'] * units.mm
#            fanal_params['blob_Eth']    = fanal_params['blob_Eth']    * units.keV
#            fanal_params['roi_Emin']    = fanal_params['roi_Emin']    * units.keV
#            fanal_params['roi_Emax']    = fanal_params['roi_Emax']    * units.keV
#        self.__init__(**fanal_params)


    def store_config(self, output_file : tb.file.File):
        output_file.set_node_attr('/FANAL', 'det_name',          self.det_name)
        output_file.set_node_attr('/FANAL', 'event_type',        self.event_type)
        output_file.set_node_attr('/FANAL', 'input_path',        self.input_path)
        output_file.set_node_attr('/FANAL', 'output_fname',      self.output_fname)
        output_file.set_node_attr('/FANAL', 'fwhm',              self.fwhm)
        output_file.set_node_attr('/FANAL', 'e_min',             self.e_min)
        output_file.set_node_attr('/FANAL', 'e_max',             self.e_max)
        output_file.set_node_attr('/FANAL', 'voxel_size_x',      self.voxel_size[0])
        output_file.set_node_attr('/FANAL', 'voxel_size_y',      self.voxel_size[1])
        output_file.set_node_attr('/FANAL', 'voxel_size_z',      self.voxel_size[2])
        output_file.set_node_attr('/FANAL', 'strict_voxel_size', self.strict_voxel_size)
        output_file.set_node_attr('/FANAL', 'voxel_Eth',         self.voxel_Eth)
        output_file.set_node_attr('/FANAL', 'veto_width',        self.veto_width)
        output_file.set_node_attr('/FANAL', 'min_veto_e',        self.min_veto_e)
        output_file.set_node_attr('/FANAL', 'track_Eth',         self.track_Eth)
        output_file.set_node_attr('/FANAL', 'max_num_tracks',    self.max_num_tracks)
        output_file.set_node_attr('/FANAL', 'blob_radius',       self.blob_radius)
        output_file.set_node_attr('/FANAL', 'blob_Eth',          self.blob_Eth)
        output_file.set_node_attr('/FANAL', 'roi_Emin',          self.roi_Emin)
        output_file.set_node_attr('/FANAL', 'roi_Emax',          self.roi_Emax)


    def run_analysis(self):
        # Print the Setup
        print(self)

        # Opening the output file and storing configration parameters
        with tb.open_file(self.output_fname, 'w', filters=tbl_filters('ZLIB4')) as output_file:
            output_file.create_group('/', 'FANAL')
            self.store_config(output_file)

        ### DATA TO COLECT
        events_data   = EventList()
        tracks_data   = TrackList()
        voxels_data   = VoxelList()
        event_counter = EventCounter()

        ### Looping through all the input files
        verbose_every    = 1
        for input_fname in self.input_fnames:

            # Updating simulated and stored event counters
            configuration_df = pd.read_hdf(input_fname, '/MC/configuration', mode='r')
            event_counter.simulated += int(configuration_df[configuration_df.param_key=='num_events'].param_value)
            event_counter.stored    += int(configuration_df[configuration_df.param_key=='saved_events'].param_value)

            # Getting event numbers
            file_event_ids = get_event_numbers_in_file(input_fname)
            print(f'* Processing {input_fname}  ({len(file_event_ids)} events) ...')

            # Getting mc hits & particles
            file_mcHits  = load_mchits_df(input_fname)
            file_mcParts = load_mcparticles_df(input_fname)

            # Looping through all the events in iFile
            for event_id in file_event_ids:

                # Updating counter of analyzed events
                event_counter.analyzed += 1
                self.logger.info(f"Analyzing event Id: {event_id} ...")

                # Analyze event
                event_data, event_tracks, event_voxels = \
                    analyze_event(self.detector, self.active_dimensions, int(event_id),
                                  self.event_type, file_mcParts.loc[event_id, :],
                                  file_mcHits.loc[event_id, :], self.sigma_Qbb, self.e_min, self.e_max, self.voxel_size, self.strict_voxel_size,
                                  self.voxel_Eth, self.veto_width, self.min_veto_e, self.track_Eth, self.max_num_tracks,
                                  self.blob_radius, self.blob_Eth, self.roi_Emin, self.roi_Emax)

                # Storing event data
                events_data.add(event_data)
                tracks_data.add(event_tracks)
                voxels_data.add(event_voxels)

                # Verbosing
                if (not(event_counter.analyzed % verbose_every)):
                    print(f'* Num analyzed events: {event_counter.analyzed}')
                if (event_counter.analyzed == (10 * verbose_every)): verbose_every *= 10


        ### STORING ANALYSIS DATA
        print(f'\n* Total analyzed events: {event_counter.analyzed}')

        # Storing events and voxels dataframes
        print(f'\n* Storing data in the output file ...\n  {output_file}\n')
        events_data.store(self.output_fname, 'FANAL')
        tracks_data.store(self.output_fname, 'FANAL')
        voxels_data.store(self.output_fname, 'FANAL')

        # Storing event counters as attributes
        events_df = events_data.df()
        event_counter.energy_filter = len(events_df[events_df.energy_filter])
        event_counter.fiduc_filter  = len(events_df[events_df.fiduc_filter])
        event_counter.track_filter  = len(events_df[events_df.track_filter])
        event_counter.blob_filter   = len(events_df[events_df.blob_filter])
        event_counter.roi_filter    = len(events_df[events_df.roi_filter])
        event_counter.store(self.output_fname, 'FANAL')

        ### Ending ...
        print('\n* Analysis done !!\n')
        print(event_counter)



# Make it executable
if __name__ == '__main__':
    try:
        config_fname = sys.argv[1]
    except IndexError:
        print("\nUsage: python nexus-production.py config_file\n")
        sys.exit()

    with open(config_fname) as config_file:
        fanal_params = json.load(config_file)
        fanal_params['fwhm']        = fanal_params['fwhm']        * units.perCent
        fanal_params['e_min']       = fanal_params['e_min']       * units.keV
        fanal_params['e_max']       = fanal_params['e_max']       * units.keV
        fanal_params['voxel_size']  = np.array(fanal_params['voxel_size'])  * units.mm
        fanal_params['voxel_Eth']   = fanal_params['voxel_Eth']   * units.keV
        fanal_params['veto_width']  = fanal_params['veto_width']  * units.mm
        fanal_params['min_veto_e']  = fanal_params['min_veto_e']  * units.keV
        fanal_params['track_Eth']   = fanal_params['track_Eth']   * units.keV
        fanal_params['blob_radius'] = fanal_params['blob_radius'] * units.mm
        fanal_params['blob_Eth']    = fanal_params['blob_Eth']    * units.keV
        fanal_params['roi_Emin']    = fanal_params['roi_Emin']    * units.keV
        fanal_params['roi_Emax']    = fanal_params['roi_Emax']    * units.keV

    fanal_setup = Setup(**fanal_params)
    fanal_setup.run_analysis()