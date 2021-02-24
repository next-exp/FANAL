# General importings
import os
import sys
import glob
import json
import numpy      as np
import tables     as tb
import pandas     as pd
from   typing import Tuple

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



def analyze(det_name          : str,
            event_type        : str,
            files_in          : str,
            file_out          : str,
            fwhm              : float,
            e_min             : float,
            e_max             : float,
            voxel_size        : Tuple[float, float, float],
            strict_voxel_size : bool,
            voxel_Eth         : float,
            veto_width        : float,
            min_veto_e        : float,
            track_Eth         : float,
            max_num_tracks    : int,
            blob_radius       : float,
            blob_Eth          : float,
            roi_Emin          : float,
            roi_Emax          : float,
            verbosity_level   : str = "WARNING"):

    """
    Parameters:
    -----------
        det_name,          # Detector name
        event_type,        # Event type: 'bb0nu', 'Tl208', 'Bi214'
        files_in,          # Input files
        file_out,          # Output file
        fwhm,              # FWHM at Qbb
        e_min,             # Minimum smeared energy for energy filtering
        e_max,             # Maximum smeared energy for energy filtering
        voxel_size,        # Voxel size (x, y, z)
        strict_voxel_size, # Force strict voxel size
        voxel_Eth,         # Voxel energy threshold
        veto_width,        # Veto width for fiducial filtering
        min_veto_e,        # Minimum energy in veto for fiducial filtering
        track_Eth,
        max_num_tracks,
        blob_radius,
        blob_Eth,
        roi_Emin,
        roi_Emax,
        verbosity_level):
    """

    ### LOGGER
    logger = get_logger('Fanal', verbosity_level)

    ### Input files
    files_in = sorted(glob.glob(files_in))

    ### Output file compression ('ZLIB1', 'ZLIB4', 'ZLIB5', 'ZLIB9', 'BLOSC5', 'BLZ4HC5')
    compression = 'ZLIB4'

    ### DETECTOR NAME & its ACTIVE dimensions
    detector          = getattr(DetName, det_name)
    ACTIVE_dimensions = get_active_size(detector)
    fid_dimensions    = get_fiducial_size(detector, veto_width)

    ### RECONSTRUCTION DATA
    # Smearing energy settings
    fwhm_Qbb  = fwhm * Qbb
    sigma_Qbb = fwhm_Qbb / 2.355
    assert e_max > e_min, 'SmE_filter settings not valid. e_max must be higher than e_min.'

    ### PRINTING GENERAL INFO
    print('\n***********************************************************************************')
    print(f'***** Detector:          {detector.name}')
    print(f'***** Reconstructing:    {event_type} events')
    print(f'***** Energy Resolution: {fwhm / units.perCent:.2f}% fwhm at Qbb')
    print(f'***** Voxel Size:        ({voxel_size[0] / units.mm}, '     + \
                                    f'{voxel_size[1] / units.mm}, '     + \
                                    f'{voxel_size[2] / units.mm}) mm  ' + \
                                    f'-  strict: {strict_voxel_size}')
    print(f'***** Voxel Eth:         {voxel_Eth/units.keV:.1f} keV')
    print(f"***** Track Eth: {track_Eth/units.keV:4.1f} keV   Max Num Tracks: {max_num_tracks}")
    print(f"***** Blob radius: {blob_radius:.1f} mm   Blob Eth: {blob_Eth/units.keV:4.1f} keV")
    print(f"***** ROI limits: [{roi_Emin/units.keV:4.1f}, {roi_Emax/units.keV:4.1f}] keV")
    print('***********************************************************************************\n')

    print(f'* Sigma at Qbb: {sigma_Qbb/units.keV:.3f} keV.\n')

    print(f'* Detector-Active dimensions [mm]:  Zmin: {ACTIVE_dimensions.z_min:7.1f}   ' + \
          f'Zmax: {ACTIVE_dimensions.z_max:7.1f}   Rmax: {ACTIVE_dimensions.rad:7.1f}')

    print(f'         ... fiducial limits [mm]:  Zmin: {fid_dimensions.z_min:7.1f}   ' + \
          f'Zmax: {fid_dimensions.z_max:7.1f}   Rmax: {fid_dimensions.rad:7.1f}\n')

    print(f'* {len(files_in)} {event_type} input files:')
    for iFileName in files_in: print(f' {iFileName}')

    print(f'\n* Output file name: {file_out}')


    ### OUTPUT FILE, ITS GROUPS & ATTRIBUTES
    # Making the output path if it does not exist
    output_path = os.path.dirname(file_out)
    if not os.path.isdir(output_path):
        print(f"  Making PATH {output_path}")
        os.makedirs(output_path)

    # Output file
    oFile = tb.open_file(file_out, 'w', filters=tbl_filters(compression))

    # Reco group Name
    oFile.create_group('/', 'FANAL')

    # Attributes
    oFile.set_node_attr('/FANAL', 'event_type',             event_type)
    oFile.set_node_attr('/FANAL', 'energy_resolution',      fwhm/units.perCent)
    oFile.set_node_attr('/FANAL', 'voxel_sizeX',            voxel_size[0])
    oFile.set_node_attr('/FANAL', 'voxel_sizeY',            voxel_size[1])
    oFile.set_node_attr('/FANAL', 'voxel_sizeZ',            voxel_size[2])
    oFile.set_node_attr('/FANAL', 'strict_voxel_size',      strict_voxel_size)
    oFile.set_node_attr('/FANAL', 'voxel_Eth',              voxel_Eth)
    oFile.set_node_attr('/FANAL', 'smE_filter_Emin',        e_min)
    oFile.set_node_attr('/FANAL', 'smE_filter_Emax',        e_max)
    oFile.set_node_attr('/FANAL', 'fiduc_filter_VetoWidth', veto_width)
    oFile.set_node_attr('/FANAL', 'fiduc_filter_MinVetoE',  min_veto_e)
    oFile.set_node_attr('/FANAL', 'track_Eth',              track_Eth)
    oFile.set_node_attr('/FANAL', 'max_num_tracks',         max_num_tracks)
    oFile.set_node_attr('/FANAL', 'blob_radius',            blob_radius)
    oFile.set_node_attr('/FANAL', 'blob_Eth',               blob_Eth)
    oFile.set_node_attr('/FANAL', 'roi_Emin',               roi_Emin)
    oFile.set_node_attr('/FANAL', 'roi_Emax',               roi_Emax)


    ### DATA TO COLECT
    events_data   = EventList()
    tracks_data   = TrackList()
    voxels_data   = VoxelList()
    event_counter = EventCounter()


    ### RECONSTRUCTION PROCEDURE

    verbose_every    = 1

    # Looping through all the input files
    for iFileName in files_in:
        # Updating simulated and stored event counters
        configuration_df = pd.read_hdf(iFileName, '/MC/configuration', mode='r')
        event_counter.simulated += int(configuration_df[configuration_df.param_key=='num_events'].param_value)
        event_counter.stored    += int(configuration_df[configuration_df.param_key=='saved_events'].param_value)

        # Getting event numbers
        file_event_ids = get_event_numbers_in_file(iFileName)
        print(f'* Processing {iFileName}  ({len(file_event_ids)} events) ...')

        # Getting mc hits & particles
        file_mcHits  = load_mchits_df(iFileName)
        file_mcParts = load_mcparticles_df(iFileName)

        # Looping through all the events in iFile
        for event_id in file_event_ids:

            # Updating counter of analyzed events
            event_counter.analyzed += 1
            logger.info(f"Analyzing event Id: {event_id} ...")

            # Analyze event
            event_data, event_tracks, event_voxels = \
                analyze_event(detector, ACTIVE_dimensions, int(event_id), event_type,
                              file_mcParts.loc[event_id, :], file_mcHits.loc[event_id, :],
                              sigma_Qbb, e_min, e_max, voxel_size, strict_voxel_size,
                              voxel_Eth, veto_width, min_veto_e, track_Eth, max_num_tracks,
                              blob_radius, blob_Eth, roi_Emin, roi_Emax)

            # Storing event data
            events_data.add(event_data)
            tracks_data.add(event_tracks)
            voxels_data.add(event_voxels)

            # Verbosing
            if (not(event_counter.analyzed % verbose_every)):
                print(f'* Num analyzed events: {event_counter.analyzed}')
            if (event_counter.analyzed == (10 * verbose_every)): verbose_every *= 10


    ### STORING RECONSTRUCTION DATA
    print(f'\n* Total analyzed events: {event_counter.analyzed}')

    # Storing events and voxels dataframes
    print(f'\n* Storing data in the output file ...\n  {file_out}\n')
    events_data.store(file_out, 'FANAL')
    tracks_data.store(file_out, 'FANAL')
    voxels_data.store(file_out, 'FANAL')

    # Storing event counters as attributes
    #event_counter.energy_filter = sum(event.energy_filter for event in events_data.events)
    #event_counter.fiduc_filter  = sum(event.fiduc_filter    for event in events_data.events)
    #event_counter.track_filter  = sum(event.track_filter  for event in events_data.events)
    #event_counter.blob_filter   = sum(event.blob_filter   for event in events_data.events)
    #event_counter.roi_filter    = sum(event.roi_filter    for event in events_data.events)

    events_df = events_data.df()
    event_counter.energy_filter = len(events_df[events_df.energy_filter])
    event_counter.fiduc_filter  = len(events_df[events_df.fiduc_filter])
    event_counter.track_filter  = len(events_df[events_df.track_filter])
    event_counter.blob_filter   = len(events_df[events_df.blob_filter])
    event_counter.roi_filter    = len(events_df[events_df.roi_filter])

    event_counter.store(file_out, 'FANAL')

    ### Ending ...
    oFile.close()
    print('\n* Analysis done !!\n')

    # Printing analysis numbers
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

    result = analyze(**fanal_params)
