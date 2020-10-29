"""
This SCRIPT runs the analysis step of FANAL.
The analysis deals with the topology and the energy of the events.
The topology will consider the number of tracks, and the blobs of the hottest track.
It loads the data from a 'reco' .h5 file and generates an .h5 file containing 2 dataFrames
that will complete the previous reco information with the new data generated in the analysis:
'events' storing all the relevant data of events
'voxels' storing all the voxels info
"""

# General importings
import sys
import tables as tb
import pandas as pd

# Specific IC stuff
import invisible_cities.core.system_of_units      as units
from invisible_cities.cities.components       import city
from invisible_cities.core.configure          import configure
from invisible_cities.reco.tbl_functions      import filters as tbl_filters

# Specific FANAL stuff
from fanal.reco.reco_io_functions import get_reco_group_name
from fanal.ana.ana_io_functions   import get_ana_group_name
from fanal.ana.ana_io_functions   import get_events_ana_dict
from fanal.ana.ana_io_functions   import extend_events_ana_dict
from fanal.ana.ana_io_functions   import store_events_ana_dict
from fanal.ana.ana_io_functions   import store_events_ana_counters
from fanal.ana.ana_io_functions   import get_voxels_ana_dict
from fanal.ana.ana_io_functions   import store_voxels_ana_dict

from fanal.ana.ana_functions      import analyze_event
from fanal.core.fanal_types       import DetName
from fanal.core.logger            import get_logger



@city
def fanal_ana(det_name,       # Detector name: 'new', 'next100', 'next500'
              event_type,     # Event type: 'bb0nu', 'Tl208', 'Bi214'
              fwhm,           # FWHM at Qbb
              voxel_size,     # Voxel size (x, y, z)
              track_Eth,      # Track energy threshold
              max_num_tracks, # Maximum number of tracks
              blob_radius,    # Blob radius
              blob_Eth,       # Blob energy threshold
              roi_Emin,       # ROI minimum energy
              roi_Emax,       # ROI maximum energy
              files_in,       # Input files
              event_range,    # Range of events to analyze: all, ... ??
              file_out,       # Output file
              compression,    # Compression of output file: 'ZLIB1', 'ZLIB4',
                              # 'ZLIB5', 'ZLIB9', 'BLOSC5', 'BLZ4HC5'
              verbosity_level):

    ### LOGGER
    logger = get_logger('FanalAna', verbosity_level)

    ### DETECTOR NAME
    det_name = getattr(DetName, det_name)


    ### PRINTING GENERAL INFO
    print('\n*******************************************************************************')
    print(f"***** Detector: {det_name.name}")
    print(f"***** Analizing {event_type} events")
    print(f"***** Energy Resolution: {fwhm/units.perCent:.2f}% FWFM at Qbb")
    print(f"***** Voxel Size: ({voxel_size[0]/units.mm}, {voxel_size[1] / units.mm}, " + \
          f"{voxel_size[2] / units.mm}) mm")
    print('*******************************************************************************\n')

    print(f"* Track Eth: {track_Eth/units.keV:4.1f} keV   Max Num Tracks: {max_num_tracks}\n")
    print(f"* Blob radius: {blob_radius:.1f} mm   Blob Eth: {blob_Eth/units.keV:4.1f} keV\n")
    print(f"* ROI limits: [{roi_Emin/units.keV:4.1f}, {roi_Emax/units.keV:4.1f}] keV\n")


    ### INPUT RECONSTRUCTION FILES AND GROUP
    reco_group_name = get_reco_group_name(fwhm/units.perCent, voxel_size)
    print(f"* {len(files_in)} {event_type} input reco file names:")
    for iFileName in files_in: print(f" {iFileName}")
    print(f"  Reco group name: {reco_group_name}\n")    


    ### OUTPUT FILE, ITS GROUPS & ATTRIBUTES
    # Output analysis file
    oFile = tb.open_file(file_out, 'w', filters=tbl_filters(compression))

    # Analysis group Name
    ana_group_name = get_ana_group_name(fwhm/units.perCent, voxel_size)
    oFile.create_group('/', 'FANAL')
    oFile.create_group('/FANAL', ana_group_name[7:])

    print(f"* Output analysis file name: {file_out}")
    print(f"  Ana group name: {ana_group_name}\n")

    # Attributes
    oFile.set_node_attr(ana_group_name, 'input_reco_files', files_in)
    oFile.set_node_attr(ana_group_name, 'input_reco_group', reco_group_name)
    oFile.set_node_attr(ana_group_name, 'event_type', event_type)
    oFile.set_node_attr(ana_group_name, 'energy_resolution', fwhm / units.perCent)
    oFile.set_node_attr(ana_group_name, 'track_Eth', track_Eth)
    oFile.set_node_attr(ana_group_name, 'max_num_tracks', max_num_tracks)
    oFile.set_node_attr(ana_group_name, 'blob_radius', blob_radius)
    oFile.set_node_attr(ana_group_name, 'blob_Eth', blob_Eth)
    oFile.set_node_attr(ana_group_name, 'roi_Emin', roi_Emin)
    oFile.set_node_attr(ana_group_name, 'roi_Emax', roi_Emax)


    ### DATA TO STORE
    # Event counters
    simulated_events     = 0
    stored_events        = 0
    smE_filter_events    = 0
    fid_filter_events    = 0
    tracks_filter_events = 0
    blobs_filter_events  = 0
    roi_filter_events    = 0

    analyzed_events = 0
    toUpdate_events = 1

    # Dictionaries for events & voxels data
    events_dict = get_events_ana_dict()
    voxels_dict = get_voxels_ana_dict()

    events_reco_df = pd.DataFrame()
    voxels_reco_df = pd.DataFrame()


    ### ANALYSIS PROCEDURE
    print('* Analyzing events ...\n')

    # Looping through all the input files
    for iFileName in files_in:
        
        # Updating reconstruction counters
        with tb.open_file(iFileName, mode='r') as reco_file:
            simulated_events  += reco_file.get_node_attr(reco_group_name, 'simulated_events')
            stored_events     += reco_file.get_node_attr(reco_group_name, 'stored_events')
            smE_filter_events += reco_file.get_node_attr(reco_group_name, 'smE_filter_events')
            fid_filter_events += reco_file.get_node_attr(reco_group_name, 'fid_filter_events')
        
        # Getting the events & voxels data from the reconstruction phase
        file_events = pd.read_hdf(iFileName, reco_group_name + '/events')
        file_voxels = pd.read_hdf(iFileName, reco_group_name + '/voxels')
        
        # Updating reconstruction dataframes
        events_reco_df = pd.concat([events_reco_df, file_events], axis=0)
        voxels_reco_df = pd.concat([voxels_reco_df, file_voxels], axis=0)
        
        print(f"* Processing {iFileName} ...")

        ### Looping through all the events that passed the fiducial filter
        for event_number, event_df in file_events[file_events.fid_filter].iterrows():        

            # Updating counter of analyzed events
            analyzed_events += 1
            logger.info(f"Analyzing event Id: {event_number} ...")

            # Analyzing event
            event_data = analyze_event(event_number, event_df,
                                       file_voxels.loc[event_number],
                                       voxels_dict,
                                       track_Eth, max_num_tracks,
                                       blob_radius, blob_Eth,
                                       roi_Emin, roi_Emax)

            # Storing event_data
            extend_events_ana_dict(events_dict, event_data)

            # Verbosing
            if (not(analyzed_events % toUpdate_events)):
                print(f"* Num analyzed events: {analyzed_events}")
            if (analyzed_events == (10 * toUpdate_events)): toUpdate_events *= 10
    
    
    ### STORING ANALYSIS DATA
    print(f"* Total analyzed events: {analyzed_events}")

    # Storing events and voxels dataframes
    print(f"\n* Storing data in the output file ...\n  {file_out}\n")
    store_events_ana_dict(file_out, ana_group_name, events_reco_df, events_dict)
    store_voxels_ana_dict(file_out, ana_group_name, voxels_reco_df, voxels_dict)

    # Storing event counters as attributes
    tracks_filter_events = sum(events_dict['tracks_filter'])
    blobs_filter_events  = sum(events_dict['blobs_filter'])
    roi_filter_events    = sum(events_dict['roi_filter'])
    
    store_events_ana_counters(oFile, ana_group_name,
                              simulated_events, stored_events,
                              smE_filter_events, fid_filter_events,
                              tracks_filter_events, blobs_filter_events,
                              roi_filter_events)
    
    
    ### Ending ...
    oFile.close()
    print('* Analysis done !!\n')
    
    print('* Event counters ...')
    print(f"  Simulated events:     {simulated_events:9}\n"     + \
          f"  Stored events:        {stored_events:9}\n"        + \
          f"  smE_filter events:    {smE_filter_events:9}\n"    + \
          f"  fid_filter events:    {fid_filter_events:9}\n"    + \
          f"  tracks_filter events: {tracks_filter_events:9}\n" + \
          f"  blobs_filter events:  {blobs_filter_events:9}\n"  + \
          f"  roi_filter events:    {roi_filter_events:9}\n")



if __name__ == '__main__':
    result = fanal_ana(**configure(sys.argv))
