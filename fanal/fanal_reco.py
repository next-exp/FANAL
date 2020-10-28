"""
This SCRIPT runs the pseudo-reconstruction step of fanalIC.
The main parameters are the energy resolution and the spatial definition.
It generates an .h5 file containing 2 dataFrames:
'events' storing all the relevant data of events
'voxels' storing all the voxels info
"""

# General importings
import sys
import tables as tb
import pandas as pd

# Specific IC stuff
import invisible_cities.core.system_of_units     as units
from invisible_cities.cities.components      import city
from invisible_cities.core.configure         import configure
from invisible_cities.reco.tbl_functions     import filters as tbl_filters
from invisible_cities.io.mcinfo_io           import get_event_numbers_in_file
from invisible_cities.io.mcinfo_io           import load_mchits_df
from invisible_cities.io.mcinfo_io           import load_mcparticles_df

# Specific fanalIC stuff
from fanal.reco.reco_io_functions            import get_reco_group_name
from fanal.reco.reco_io_functions            import get_events_reco_dict
from fanal.reco.reco_io_functions            import extend_events_reco_dict
from fanal.reco.reco_io_functions            import store_events_reco_dict
from fanal.reco.reco_io_functions            import store_events_reco_counters
from fanal.reco.reco_io_functions            import get_voxels_reco_dict
from fanal.reco.reco_io_functions            import store_voxels_reco_dict
from fanal.reco.reco_functions               import reconstruct_event

from fanal.core.logger                       import get_logger
from fanal.core.detector                     import get_active_size
from fanal.core.detector                     import get_fiducial_size
from fanal.core.fanal_types                  import DetName



### GENERAL DATA NEEDED
Qbb  = 2457.83 * units.keV
DRIFT_VELOCITY = 1. * units.mm / units.mus


@city
def fanal_reco(det_name,    # Detector name: 'new', 'next100', 'next500'
               event_type,  # Event type: 'bb0nu', 'Tl208', 'Bi214'
               fwhm,        # FWHM at Qbb
               e_min,       # Minimum smeared energy for energy filtering
               e_max,       # Maximum smeared energy for energy filtering
               voxel_size,  # Voxel size (x, y, z)
               voxel_Eth,   # Voxel energy threshold
               veto_width,  # Veto width for fiducial filtering
               min_veto_e,  # Minimum energy in veto for fiducial filtering
               files_in,    # Input files
               event_range, # Range of events to analyze: all, ... ??
               file_out,    # Output file
               compression, # Compression of output file: 'ZLIB1', 'ZLIB4',
                            # 'ZLIB5', 'ZLIB9', 'BLOSC5', 'BLZ4HC5'
               verbosity_level):


    ### LOGGER
    logger = get_logger('FanalReco', verbosity_level)


    ### DETECTOR NAME & its ACTIVE dimensions
    det_name          = getattr(DetName, det_name)
    ACTIVE_dimensions = get_active_size(det_name)
    fid_dimensions    = get_fiducial_size(det_name, veto_width)


    ### RECONSTRUCTION DATA
    # Smearing energy settings
    fwhm_Qbb  = fwhm * Qbb
    sigma_Qbb = fwhm_Qbb / 2.355
    assert e_max > e_min, 'SmE_filter settings not valid. e_max must be higher than e_min.'


    ### PRINTING GENERAL INFO
    print('\n*******************************************************************************')
    print(f'***** Detector:          {det_name.name}')
    print(f'***** Reconstructing:    {event_type} events')
    print(f'***** Energy Resolution: {fwhm / units.perCent:.2f}% fwhm at Qbb')
    print(f'***** Voxel Size:        ({voxel_size[0] / units.mm}, ' + \
                                    f'{voxel_size[1] / units.mm}, ' + \
                                    f'{voxel_size[2] / units.mm}) mm')
    print(f'***** Voxel Eth:         {voxel_Eth/units.keV:.1f} keV')
    print('*******************************************************************************\n')

    print(f'* Sigma at Qbb: {sigma_Qbb/units.keV:.3f} keV.\n')

    print(f'* Detector-Active dimensions [mm]:  Zmin: {ACTIVE_dimensions.z_min:7.1f}   ' + \
          f'Zmax: {ACTIVE_dimensions.z_max:7.1f}   Rmax: {ACTIVE_dimensions.rad:7.1f}')
    
    print(f'         ... fiducial limits [mm]:  Zmin: {fid_dimensions.z_min:7.1f}   ' + \
          f'Zmax: {fid_dimensions.z_max:7.1f}   Rmax: {fid_dimensions.rad:7.1f}\n')

    print(f'* {len(files_in)} {event_type} input files:')
    for iFileName in files_in: print(f' {iFileName}')


    ### OUTPUT FILE, ITS GROUPS & ATTRIBUTES
    # Output file
    oFile = tb.open_file(file_out, 'w', filters = tbl_filters(compression))

    # Reco group Name
    reco_group_name = get_reco_group_name(fwhm/units.perCent, voxel_size)
    oFile.create_group('/', 'FANAL')
    oFile.create_group('/FANAL', reco_group_name[7:])

    print(f'\n* Output file name: {file_out}')
    print(f'  Reco group name:  {reco_group_name}\n')

    # Attributes
    oFile.set_node_attr(reco_group_name, 'input_sim_files',           files_in)
    oFile.set_node_attr(reco_group_name, 'event_type',                event_type)
    oFile.set_node_attr(reco_group_name, 'energy_resolution',         fwhm/units.perCent)
    oFile.set_node_attr(reco_group_name, 'voxel_sizeX',               voxel_size[0])
    oFile.set_node_attr(reco_group_name, 'voxel_sizeY',               voxel_size[1])
    oFile.set_node_attr(reco_group_name, 'voxel_sizeZ',               voxel_size[2])
    oFile.set_node_attr(reco_group_name, 'voxel_Eth',                 voxel_Eth)
    oFile.set_node_attr(reco_group_name, 'smE_filter_Emin',           e_min)
    oFile.set_node_attr(reco_group_name, 'smE_filter_Emax',           e_max)
    oFile.set_node_attr(reco_group_name, 'fiducial_filter_VetoWidth', veto_width)
    oFile.set_node_attr(reco_group_name, 'fiducial_filter_MinVetoE',  min_veto_e)


    ### DATA TO STORE
    # Event counters
    simulated_events = 0
    stored_events    = 0
    analyzed_events  = 0
    toUpdate_events  = 1

    # Dictionaries for events & voxels data
    events_dict = get_events_reco_dict()
    voxels_dict = get_voxels_reco_dict()


    ### RECONSTRUCTION PROCEDURE
    # Looping through all the input files
    for iFileName in files_in:
        # Updating simulated and stored event counters
        configuration_df  = pd.read_hdf(iFileName, '/MC/configuration', mode='r')
        simulated_events += int(configuration_df[configuration_df.param_key == 'num_events'].param_value)
        stored_events    += int(configuration_df[configuration_df.param_key == 'saved_events'].param_value)

        # Getting event numbers
        file_event_numbers = get_event_numbers_in_file(iFileName)
        print(f'* Processing {iFileName}  ({len(file_event_numbers)} events) ...')

        # Getting mc hits & particles
        file_mcHits  = load_mchits_df(iFileName)
        file_mcParts = load_mcparticles_df(iFileName)

        # Looping through all the events in iFile
        for event_number in file_event_numbers:

            # Updating counter of analyzed events
            analyzed_events += 1
            logger.info(f"Reconstructing event Id: {event_number} ...")

            # Reconstructing event
            event_data = reconstruct_event(det_name, ACTIVE_dimensions,
                                           event_number, event_type,
                                           sigma_Qbb, e_min, e_max,
                                           voxel_size, voxel_Eth,
                                           veto_width, min_veto_e,
                                           file_mcParts.loc[event_number, :],
                                           file_mcHits .loc[event_number, :],
                                           voxels_dict)

            # Storing event_data
            extend_events_reco_dict(events_dict, event_data)

            # Verbosing
            if (not(analyzed_events % toUpdate_events)):
                print(f'* Num analyzed events: {analyzed_events}')
            if (analyzed_events == (10 * toUpdate_events)): toUpdate_events *= 10
            

    ### STORING RECONSTRUCTION DATA
    print(f'* Total analyzed events: {analyzed_events}')

    # Storing events and voxels dataframes
    print(f'\n* Storing data in the output file ...\n  {file_out}\n')
    store_events_reco_dict(file_out, reco_group_name, events_dict)
    store_voxels_reco_dict(file_out, reco_group_name, voxels_dict)

    # Storing event counters as attributes
    smE_filter_events = sum(events_dict['smE_filter'])
    fid_filter_events = sum(events_dict['fid_filter'])
    store_events_reco_counters(oFile, reco_group_name, simulated_events,
                               stored_events, smE_filter_events, fid_filter_events)

    oFile.close()
    print('* Reconstruction done !!\n')

    # Printing reconstruction numbers
    print('* Event counters ...')
    print(f"  Simulated events:  {simulated_events:9}\n"  + \
          f"  Stored events:     {stored_events:9}\n"     + \
          f"  smE_filter events: {smE_filter_events:9}\n" + \
          f"  fid_filter events: {fid_filter_events:9}\n")


if __name__ == '__main__':
	result = fanal_reco(**configure(sys.argv))
