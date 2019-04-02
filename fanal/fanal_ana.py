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
import invisible_cities.core.system_of_units as units
from invisible_cities.cities.components       import city
from invisible_cities.core.configure          import configure
from invisible_cities.evm.event_model         import Voxel
from invisible_cities.reco.tbl_functions      import filters as tbl_filters
from invisible_cities.reco.paolina_functions  import make_track_graphs
from invisible_cities.reco.paolina_functions  import length as track_length
from invisible_cities.reco.paolina_functions  import blob_energies

# Specific fanalIC stuff
from fanal.reco.reco_io_functions import get_reco_group_name
from fanal.ana.ana_io_functions   import get_ana_group_name
from fanal.ana.ana_io_functions   import get_events_ana_dict
from fanal.ana.ana_io_functions   import get_voxels_ana_dict
from fanal.ana.ana_io_functions   import extend_events_ana_data
from fanal.ana.ana_io_functions   import extend_voxels_ana_data
from fanal.ana.ana_io_functions   import store_events_ana_data
from fanal.ana.ana_io_functions   import store_voxels_ana_data
from fanal.ana.ana_io_functions   import store_events_ana_counters

from fanal.core.fanal_types  import DetName
from fanal.core.fanal_types  import SpatialDef

from fanal.core.logger       import get_logger

from fanal.ana.ana_functions import get_new_energies
from fanal.ana.ana_functions import get_voxel_track_relations
from fanal.ana.ana_functions import process_tracks



@city
def fanal_ana(det_name,       # Detector name: 'new', 'next100', 'next500'
              event_type,     # Event type: 'bb0nu', 'Tl208', 'Bi214'
              fwhm,           # FWHM at Qbb
              spatial_def,    # Spatial definition: 'low', 'high'
              voxel_Eth,      # Voxel energy threshold
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

    ### SPATIAL DEFINITION
    spatial_def = getattr(SpatialDef, spatial_def)


    ### PRINTING GENERAL INFO
    print('\n***********************************************************************************')
    print('***** Detector: {}'.format(det_name.name))
    print('***** Analizing {} events'.format(event_type))
    print('***** Energy Resolution: {:.2f}% FWFM at Qbb'.format(fwhm / units.perCent))
    print('***** Spatial definition: {}'.format(spatial_def.name))
    print('***********************************************************************************\n')

    print('* Voxel Eth: {:4.1f} keV   Track Eth: {:4.1f} keV   Max Num Tracks: {}\n'
          .format(voxel_Eth/units.keV, track_Eth/units.keV, max_num_tracks))
    print('* Blob radius: {:.1f} mm   Blob Eth: {:4.1f} keV\n'
          .format(blob_radius, blob_Eth / units.keV))
    print('* ROI limits: [{:4.1f}, {:4.1f}] keV\n'
          .format(roi_Emin/units.keV, roi_Emax/units.keV))


    ### INPUT RECONSTRUCTION FILE AND GROUP
    reco_group_name = get_reco_group_name(fwhm/units.perCent, spatial_def)
    print('* Input reco file name:', files_in)
    print('  Reco group name: {}\n'.format(reco_group_name))


    ### OUTPUT FILE, ITS GROUPS & ATTRIBUTES
    # Output analysis file
    oFile = tb.open_file(file_out, 'w', filters=tbl_filters(compression))

    # Analysis group Name
    ana_group_name = get_ana_group_name(fwhm/units.perCent, spatial_def)
    oFile.create_group('/', 'FANALIC')
    oFile.create_group('/FANALIC', ana_group_name[9:])

    print('* Output analysis file name:', file_out)
    print('  Ana group name: {}\n'.format(ana_group_name))

    # Attributes
    oFile.set_node_attr(ana_group_name, 'input_reco_file', files_in[0])
    oFile.set_node_attr(ana_group_name, 'input_reco_group', reco_group_name)
    oFile.set_node_attr(ana_group_name, 'event_type', event_type)
    oFile.set_node_attr(ana_group_name, 'energy_resolution', fwhm / units.perCent)
    oFile.set_node_attr(ana_group_name, 'voxel_Eth', voxel_Eth)
    oFile.set_node_attr(ana_group_name, 'track_Eth', track_Eth)
    oFile.set_node_attr(ana_group_name, 'max_num_tracks', max_num_tracks)
    oFile.set_node_attr(ana_group_name, 'blob_radius', blob_radius)
    oFile.set_node_attr(ana_group_name, 'blob_Eth', blob_Eth)
    oFile.set_node_attr(ana_group_name, 'roi_Emin', roi_Emin)
    oFile.set_node_attr(ana_group_name, 'roi_Emax', roi_Emax)


    ### DATA TO STORE
    # Dictionaries for events & voxels data
    events_dict = get_events_ana_dict()
    voxels_dict = get_voxels_ana_dict()


    ### ANALYSIS PROCEDURE

    # Getting the events & voxels data from the reconstruction phase
    # This is the option a little bit slower that requires less memory ... (TO BE CHECKED!!)
    # event_numbers_toAnalize = pd.read_hdf(files_in[0], reco_group_name + '/events',
    #                                      where=['fid_filter = True']).index
    # And this is the fastest option requiring more memory
    events_df = pd.read_hdf(files_in[0], reco_group_name + '/events')
    voxels_df = pd.read_hdf(files_in[0], reco_group_name + '/voxels')

    # Identifying as negligible all the voxels with energy lower than threshold
    voxels_df['negli'] = voxels_df.E < voxel_Eth
    print('* Total Voxels in File: {0}     Negligible Voxels (below {1:3.1f} keV): {2}\n'
          .format(len(voxels_df), voxel_Eth / units.keV,
                  len(voxels_df[voxels_df.negli == True])))

    # Analyzing only the fiducial events ...
    print('* Analyzing events ...\n')

    # Counter of analyzed events for verbosing pourpouses
    num_analyzed_events = 0

    # Looping through all the events that passed the fiducial filter
    for event_id in events_df[events_df.fid_filter].index:

        # Updating counter of analyzed events
        num_analyzed_events += 1
        if not int(str(num_analyzed_events)[-int(math.log10(num_analyzed_events)):]):
            print('* Num analyzed events: {}'.format(num_analyzed_events))

        # Verbosing
        logger.info('Analyzing event Id: {0} ...'.format(event_id))

        # Getting the voxels of current event and their sizes
        event_voxels = voxels_df[voxels_df.event_id == event_id]
        num_event_voxels = len(event_voxels)
        num_event_voxels_negli = len(event_voxels[event_voxels.negli == True])
        event_data = events_df.loc[event_id]
        voxel_dimensions = (event_data.voxel_sizeX, event_data.voxel_sizeY,
                            event_data.voxel_sizeZ)

        logger.info('  Total Voxels: {}   Negli. Voxels: {}   Voxels Size: ({:3.1f}, {:3.1f}, {:3.1f}) mm'
                    .format(num_event_voxels, num_event_voxels_negli,
                            voxel_dimensions[0], voxel_dimensions[1],
                            voxel_dimensions[2]))

        # If there is any negligible Voxel,
        # distribute its energy between its neighbours,
        # if not, all voxels maintain their previous energies
        if num_event_voxels_negli:
            event_voxels_newE = get_new_energies(event_voxels)
        else:
            event_voxels_newE = event_voxels.E.tolist()

        # Translate fanalIC voxels info to IC voxels to make tracks
        ic_voxels = [Voxel(event_voxels.iloc[i].X, event_voxels.iloc[i].Y,
                           event_voxels.iloc[i].Z, event_voxels_newE[i],
                           voxel_dimensions) for i in range(num_event_voxels)]

        # Make tracks
        event_tracks = make_track_graphs(ic_voxels)
        num_event_tracks = len(event_tracks)
        logger.info('  Num initial tracks: {:2}'.format(num_event_tracks))

        # Appending to every voxel, the track it belongs to
        event_voxels_tracks = get_voxel_track_relations(event_voxels, event_tracks)

        # Appending info of this event voxels
        extend_voxels_ana_data(voxels_dict, event_voxels.index,
                               event_voxels_newE, event_voxels_tracks)

        # Processing tracks: Getting energies, sorting and filtering ...
        event_sorted_tracks = process_tracks(event_tracks, track_Eth)
        num_event_tracks    = len(event_sorted_tracks)

        # Storing 3 hottest track info
        if num_event_tracks >= 1:
            track0_E      = event_sorted_tracks[0][0]
            track0_voxels = len(event_sorted_tracks[0][1].nodes())
            track0_length = track_length(event_sorted_tracks[0][1])
        else:
            track0_E = track0_voxels = track0_length = np.nan
        if num_event_tracks >= 2:
            track1_E      = event_sorted_tracks[1][0]
            track1_voxels = len(event_sorted_tracks[1][1].nodes())
            track1_length = track_length(event_sorted_tracks[1][1])
        else:
            track1_E = track1_voxels = track1_length = np.nan
        if num_event_tracks >= 3:
            track2_E      = event_sorted_tracks[2][0]
            track2_voxels = len(event_sorted_tracks[2][1].nodes())
            track2_length = track_length(event_sorted_tracks[2][1])
        else:
            track2_E = track2_voxels = track2_length = np.nan

        # Applying the tracks filter
        tracks_filter = ((num_event_tracks > 0) &
                         (num_event_tracks <= max_num_tracks))

        # Verbosing
        logger.info('  Num final tracks: {:2}  -->  tracks_filter: {}'
                    .format(num_event_tracks, tracks_filter))

        # For those events NOT passing the tracks filter:
        # Storing data of NON tracks_filter vents
        if not tracks_filter:
            extend_events_ana_data(events_dict, event_id, num_event_tracks,
                                   track0_E, track0_voxels, track0_length,
                                   track1_E, track1_voxels, track1_length,
                                   track2_E, track2_voxels, track2_length,
                                   tracks_filter)

        # Only for those events passing the tracks filter:
        else:
            # Getting the blob energies of the track with highest energy
            blob1_E, blob2_E = blob_energies(event_sorted_tracks[0][1], blob_radius)

            # Applying the blobs filter
            blobs_filter = (blob2_E > blob_Eth)

            # Verbosing
            logger.info('  Blob 1 energy: {:4.1f} keV   Blob 2 energy: {:4.1f} keV  -->  Blobs filter: {}'
                        .format(blob1_E/units.keV, blob2_E/units.keV, blobs_filter))

            # For those events NOT passing the blobs filter:
            # Storing data of NON blobs_filter vents
            if not blobs_filter:
                extend_events_ana_data(events_dict, event_id, num_event_tracks,
                                       track0_E, track0_voxels, track0_length,
                                       track1_E, track1_voxels, track1_length,
                                       track2_E, track2_voxels,  track2_length,
                                       tracks_filter, blob1_E = blob1_E,
                                       blob2_E = blob2_E, blobs_filter = blobs_filter)

            # Only for those events passing the blobs filter:
            else:
                # Getting the total event smeared energy
                event_smE = events_df.loc[event_id].smE

                # Applying the ROI filter
                roi_filter = ((event_smE >= roi_Emin) & (event_smE <= roi_Emax))

                # Verbosing
                logger.info('  Event energy: {:6.1f} keV  -->  ROI filter: {}'
                            .format(event_smE / units.keV, roi_filter))

                # Storing all the events (as this is the last filter)
                extend_events_ana_data(events_dict, event_id, num_event_tracks,
                                       track0_E, track0_voxels, track0_length,
                                       track1_E, track1_voxels, track1_length,
                                       track2_E, track2_voxels, track2_length,
                                       tracks_filter, blob1_E = blob1_E,
                                       blob2_E = blob2_E, blobs_filter = blobs_filter,
                                       roi_filter = roi_filter)


    ### STORING DATA
    # Storing events and voxels dataframes
    print('\n* Storing data in the output file ...\n  {}\n'.format(file_out))
    store_events_ana_data(file_out, ana_group_name, events_df, events_dict)
    store_voxels_ana_data(file_out, ana_group_name, voxels_df, voxels_dict)

    # Storing event counters as attributes
    tracks_filter_events, blobs_filter_events, roi_filter_events = \
        store_events_ana_counters(oFile, ana_group_name, events_df)


    ### Ending ...
    oFile.close()
    print('* Analysis done !!\n')

    # Printing analysis numbers
    with tb.open_file(files_in[0], mode='r') as iFile:
        simulated_events  = iFile.get_node_attr(reco_group_name, 'simulated_events')
        stored_events     = iFile.get_node_attr(reco_group_name, 'stored_events')
        smE_filter_events = iFile.get_node_attr(reco_group_name, 'smE_filter_events')
        fid_filter_events = iFile.get_node_attr(reco_group_name, 'fid_filter_events')

    print('''* Event counters:
  Simulated events:     {0:9}
  Stored events:        {1:9}
  smE_filter events:    {2:9}
  fid_filter events:    {3:9}
  tracks_filter events: {4:9}
  blobs_filter events:  {5:9}
  roi_filter events:    {6:9}\n'''
    .format(simulated_events, stored_events, smE_filter_events,
            fid_filter_events, tracks_filter_events, blobs_filter_events,
            roi_filter_events))



if __name__ == '__main__':
    result = fanal_ana(**configure(sys.argv))
