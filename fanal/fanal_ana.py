"""
This SCRIPT runs the analysis step of fanalIC.
The analysis deals with the topology and the energy of the events.
The topology will consider the number of tracks, and the blobs of the hottest track.
It loads the data from a 'reco' .h5 file and generates an .h5 file containing 2 dataFrames
that will complete the previous reco information with the new data generated in the analysis:
'events' storing all the relevant data of events
'voxels' storing all the voxels info
"""

# General importings
import os
import sys
import math
import numpy  as np
import tables as tb
import pandas as pd

# Specific IC stuff
import invisible_cities.core.system_of_units  as units
from invisible_cities.cities.components       import city
from invisible_cities.core.configure          import configure
from invisible_cities.evm.event_model         import Voxel
from invisible_cities.reco.tbl_functions      import filters as tbl_filters
from invisible_cities.reco.paolina_functions  import make_track_graphs
from invisible_cities.reco.paolina_functions  import blob_energies

# Specific fanalIC stuff
from fanal.reco.reco_io_functions import get_reco_group_name
from fanal.ana.ana_io_functions   import get_ana_group_name
from fanal.ana.ana_io_functions   import get_event_ana_data
from fanal.ana.ana_io_functions   import get_events_ana_dict
from fanal.ana.ana_io_functions   import extend_events_ana_dict
from fanal.ana.ana_io_functions   import store_events_ana_dict
from fanal.ana.ana_io_functions   import store_events_ana_counters
from fanal.ana.ana_io_functions   import get_voxels_ana_dict
from fanal.ana.ana_io_functions   import extend_voxels_ana_dict
from fanal.ana.ana_io_functions   import store_voxels_ana_dict

from fanal.core.fanal_types  import DetName
from fanal.core.logger       import get_logger

from fanal.ana.ana_functions import get_new_energies
from fanal.ana.ana_functions import get_voxel_track_relations
from fanal.ana.ana_functions import process_tracks



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
    print('\n***********************************************************************************')
    print('***** Detector: {}'.format(det_name.name))
    print('***** Analizing {} events'.format(event_type))
    print('***** Energy Resolution: {:.2f}% FWFM at Qbb'.format(fwhm / units.perCent))
    print('***** Voxel Size: ({}, {}, {}) mm'.format(voxel_size[0] / units.mm,
                                                     voxel_size[1] / units.mm,
                                                     voxel_size[2] / units.mm))
    print('***********************************************************************************\n')

    print('* Track Eth: {:4.1f} keV   Max Num Tracks: {}\n'
          .format(track_Eth/units.keV, max_num_tracks))
    print('* Blob radius: {:.1f} mm   Blob Eth: {:4.1f} keV\n'
          .format(blob_radius, blob_Eth / units.keV))
    print('* ROI limits: [{:4.1f}, {:4.1f}] keV\n'
          .format(roi_Emin/units.keV, roi_Emax/units.keV))


    ### INPUT RECONSTRUCTION FILE AND GROUP
    reco_group_name = get_reco_group_name(fwhm/units.perCent, voxel_size)
    print('* {} {} input reco file names:'.format(len(files_in), event_type))
    for iFileName in files_in: print(' ', iFileName)
    print('  Reco group name: {}\n'.format(reco_group_name))    


    ### OUTPUT FILE, ITS GROUPS & ATTRIBUTES
    # Output analysis file
    oFile = tb.open_file(file_out, 'w', filters=tbl_filters(compression))

    # Analysis group Name
    ana_group_name = get_ana_group_name(fwhm/units.perCent, voxel_size)
    oFile.create_group('/', 'FANAL')
    oFile.create_group('/FANAL', ana_group_name[9:])

    print('* Output analysis file name:', file_out)
    print('  Ana group name: {}\n'.format(ana_group_name))

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
        
        print('* Processing {} ...'.format(iFileName))

        ### Looping through all the events that passed the fiducial filter
        for event_number, event_df in file_events[file_events.fid_filter].iterrows():        

            # Updating counter of analyzed events
            analyzed_events += 1
            logger.info('Analyzing event Id: {0} ...'.format(event_number))

            # Getting event data
            event_data = get_event_ana_data()
            event_data['event_id'] = event_number
            
            event_voxels = file_voxels.loc[event_number]
            num_event_voxels = len(event_voxels)
            num_event_voxels_negli = len(event_voxels[event_voxels.negli])
            voxel_dimensions = (event_df.voxel_sizeX,
                                event_df.voxel_sizeY,
                                event_df.voxel_sizeZ)

            logger.info('  Total Voxels: {}   Negli. Voxels: {}   Voxels Size: ({:3.1f}, {:3.1f}, {:3.1f}) mm'\
                        .format(num_event_voxels, num_event_voxels_negli, voxel_dimensions[0],
                                voxel_dimensions[1], voxel_dimensions[2]))

            # If there is any negligible Voxel, distribute its energy between its neighbours,
            # if not, all voxels maintain their previous energies
            if num_event_voxels_negli:
                event_voxels_newE = get_new_energies(event_voxels)
            else:
                event_voxels_newE = event_voxels.E.tolist()
        
            # Translate fanalIC voxels info to IC voxels to make tracks
            #ic_voxels = [Voxel(event_voxels.iloc[i].X, event_voxels.iloc[i].Y, event_voxels.iloc[i].Z,
            #                   event_voxels_newE[i], voxel_dimensions) for i in range(num_event_voxels)]
            ic_voxels = []
            for i, voxel in event_voxels.iterrows():
                ic_voxel = Voxel(voxel.X, voxel.Y, voxel.Z, event_voxels_newE[i], voxel_dimensions)
                ic_voxels.append(ic_voxel)            
            
            # Make tracks
            event_tracks = make_track_graphs(ic_voxels)
            num_ini_tracks = len(event_tracks)
            logger.info('  Num initial tracks: {:2}'.format(num_ini_tracks))

            # Appending to every voxel, the track it belongs to
            event_voxels_tracks = get_voxel_track_relations(event_voxels, event_tracks)

            # Appending ana-info to this event voxels
            extend_voxels_ana_dict(voxels_dict, event_number, event_voxels.index.tolist(),
                                   event_voxels_newE, event_voxels_tracks)

            # Processing tracks: Getting energies, sorting and filtering ...
            event_sorted_tracks = process_tracks(event_tracks, track_Eth)         
            event_data['num_tracks'] = len(event_sorted_tracks)

            # Getting 3 hottest tracks info
            if event_data['num_tracks'] >= 1:
                event_data['track0_E']      = event_sorted_tracks[0][0]
                event_data['track0_length'] = event_sorted_tracks[0][1]
                event_data['track0_voxels'] = len(event_sorted_tracks[0][2].nodes())
            if event_data['num_tracks'] >= 2:
                event_data['track1_E']      = event_sorted_tracks[1][0]
                event_data['track1_length'] = event_sorted_tracks[1][1]
                event_data['track1_voxels'] = len(event_sorted_tracks[1][2].nodes())
            if event_data['num_tracks'] >= 3:
                event_data['track2_E']      = event_sorted_tracks[2][0]
                event_data['track2_length'] = event_sorted_tracks[2][1]
                event_data['track2_voxels'] = len(event_sorted_tracks[2][2].nodes())
            
            # Applying the tracks filter consisting on:
            # 0 < num tracks < max_num_tracks
            # the track length must be longer than 2 times the blob_radius
            event_data['tracks_filter'] = ((event_data['num_tracks'] >  0) &
                                           (event_data['num_tracks'] <= max_num_tracks) &
                                           (event_data['track0_length'] >=  2. * blob_radius))
       
            # Verbosing
            logger.info('  Num final tracks: {:2}  -->  tracks_filter: {}' \
                        .format(event_data['num_tracks'], event_data['tracks_filter']))

            
            ### For those events passing the tracks filter:
            if event_data['tracks_filter']:

                # Getting the blob energies of the track with highest energy
                event_data['blob1_E'], event_data['blob2_E'] = \
                    blob_energies(event_sorted_tracks[0][2], blob_radius)
                
                # Applying the blobs filter
                event_data['blobs_filter'] = (event_data['blob2_E'] > blob_Eth)
               
                # Verbosing
                logger.info('  Blob 1 energy: {:4.1f} keV   Blob 2 energy: {:4.1f} keV  -->  Blobs filter: {}'\
                            .format(event_data['blob1_E']/units.keV, event_data['blob2_E']/units.keV,
                                    event_data['blobs_filter']))

                
                ### For those events passing the blobs filter:
                if event_data['blobs_filter']:

                    # Getting the total event smeared energy
                    event_smE = file_events.loc[event_number].smE
                    
                    # Applying the ROI filter
                    event_data['roi_filter'] = ((event_smE >= roi_Emin) & (event_smE <= roi_Emax))
                
                    # Verbosing
                    logger.info('  Event energy: {:6.1f} keV  -->  ROI filter: {}'\
                                .format(event_smE / units.keV, event_data['roi_filter']))  

                    
            # Storing event_data
            extend_events_ana_dict(events_dict, event_data)

            # Verbosing
            if (not(analyzed_events % toUpdate_events)):
                print('* Num analyzed events: {}'.format(analyzed_events))
            if (analyzed_events == (10 * toUpdate_events)): toUpdate_events *= 10
    
    
    ### STORING ANALYSIS DATA
    print('* Total analyzed events: {}'.format(analyzed_events))

    # Storing events and voxels dataframes
    print('\n* Storing data in the output file ...\n  {}\n'.format(file_out))
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
    
    print('''* Event counters:
  Simulated events:     {0:9}
  Stored events:        {1:9}
  smE_filter events:    {2:9}
  fid_filter events:    {3:9}
  tracks_filter events: {4:9}
  blobs_filter events:  {5:9}
  roi_filter events:    {6:9}'''
          .format(simulated_events, stored_events, smE_filter_events,
                  fid_filter_events, tracks_filter_events, blobs_filter_events,
                  roi_filter_events))



if __name__ == '__main__':
    result = fanal_ana(**configure(sys.argv))
