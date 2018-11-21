"""
This SCRIPT runs the analysis step of fanalIC.
The analysis deals with the topology and the energy of the events.
The topology will consider the number of tracks, and the blobs of the hottest track.
It loads the data from a 'reco' .h5 file and generates an .h5 file containing 2 dataFrames
that will complete the previous reco information with the new data generated in the analysis:
'events' storing all the relevant data of events
'voxels' storing all the voxels info
"""


#---- imports

import os
import sys
import math
import numpy  as np
import tables as tb
import pandas as pd
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm

from operator import itemgetter

# Specific IC stuff
from invisible_cities.core.system_of_units_c  import units
from invisible_cities.reco.paolina_functions  import make_track_graphs
from invisible_cities.reco.paolina_functions  import blob_energies
from invisible_cities.evm.event_model         import Voxel

# Specific fanalIC stuff
from fanal.reco.reco_io_functions import get_reco_file_name
from fanal.reco.reco_io_functions import get_reco_group_name
from fanal.ana.ana_io_functions import get_ana_file_name
from fanal.ana.ana_io_functions import get_ana_group_name
from fanal.ana.ana_io_functions import get_events_ana_dict
from fanal.ana.ana_io_functions import get_voxels_ana_dict
from fanal.ana.ana_io_functions import extend_events_ana_data
from fanal.ana.ana_io_functions import extend_voxels_ana_data
from fanal.ana.ana_io_functions import store_events_ana_data
from fanal.ana.ana_io_functions import store_voxels_ana_data
from fanal.ana.ana_io_functions import store_events_ana_counters

from fanal.core.mc_utilities import print_mc_event
from fanal.core.mc_utilities import plot_mc_event

from fanal.ana.ana_functions import get_new_energies
from fanal.ana.ana_functions import get_voxel_track_relations
from fanal.ana.ana_functions import process_tracks



#--- Configuration

# GENERAL 
VERBOSITY_LEVEL = 1
EVENT_TYPE = 'Bi214'
DET_NAME = 'NEXT100'

# PSEUDO-RECONSTRUCTION
FWHM_Qbb_perc = 0.7 * units.perCent
SPATIAL_DEFINITION = 'Std'

# CURRENT TOPOLOGY ANALYSYS
VOXEL_E_THRESHOLD = 2 * units.keV
TRACK_E_THRESHOLD = 10 * units.keV
MAX_NUM_TRACKS = 1
BLOB_RADIUS = 18 * units.mm
BLOB_E_THRESHOLD = 350 * units.keV

# CURRENT ROI ANALYSYS
ROI_E_MIN = 2453 * units.keV
ROI_E_MAX = 2475 * units.keV


print('\n***********************************************************************************')
print('***** Detector: {}'.format(DET_NAME))
print('***** Analizing {} events'.format(EVENT_TYPE))
print('***** Energy Resolution: {:.2f}% FWFM at Qbb'.format(FWHM_Qbb_perc / units.perCent))
print('***** Spatial definition: {}'.format(SPATIAL_DEFINITION))
print('***********************************************************************************')

if (VERBOSITY_LEVEL >= 1):
    print('\n* Voxel Eth: {:4.1f} keV   Track Eth: {:4.1f} keV   Max Num Tracks: {}' \
          .format(VOXEL_E_THRESHOLD/units.keV, TRACK_E_THRESHOLD/units.keV, MAX_NUM_TRACKS))
    print('\n* Blob radius: {:.1f} mm   Blob Eth: {:4.1f} keV' \
          .format(BLOB_RADIUS, BLOB_E_THRESHOLD / units.keV))
    print('\n* ROI limits: [{:4.1f}, {:4.1f}] keV'.format(ROI_E_MIN/units.keV, ROI_E_MAX/units.keV))



#--- Input files

RECO_PATH = '/Users/Javi/Development/fanalIC_NB/data/reco/'
iFileName = get_reco_file_name(RECO_PATH, EVENT_TYPE)
reco_group_name = get_reco_group_name(FWHM_Qbb_perc/units.perCent, SPATIAL_DEFINITION)

if (VERBOSITY_LEVEL >= 1):
    print('\n* Input reco file name:', iFileName)
    print('  Reco group name:', reco_group_name)



#--- Ouput files and group

ANA_PATH = '/Users/Javi/Development/fanalIC_NB/data/ana/'
oFileName = get_ana_file_name(ANA_PATH, EVENT_TYPE)
ana_group_name = get_ana_group_name(FWHM_Qbb_perc/units.perCent, SPATIAL_DEFINITION)

if (VERBOSITY_LEVEL >= 1):
    print('\n* Output analysis file name:', oFileName)
    print('  Ana group name:', ana_group_name)

# Creating the output files and its groups
oFile_filters = tb.Filters(complib='zlib', complevel=4)
oFile = tb.open_file(oFileName, 'w', filters=oFile_filters)
oFile.create_group('/', 'FANALIC')
oFile.create_group('/FANALIC', ana_group_name[9:])

# Storing all the parameters of current analysis
# as attributes of the ana_group
oFile.set_node_attr(ana_group_name, 'input_reco_file', iFileName)
oFile.set_node_attr(ana_group_name, 'input_reco_group', reco_group_name)
oFile.set_node_attr(ana_group_name, 'event_type', EVENT_TYPE)
oFile.set_node_attr(ana_group_name, 'energy_resolution', FWHM_Qbb_perc/units.perCent)
oFile.set_node_attr(ana_group_name, 'voxel_Eth', VOXEL_E_THRESHOLD)
oFile.set_node_attr(ana_group_name, 'track_Eth', TRACK_E_THRESHOLD)
oFile.set_node_attr(ana_group_name, 'max_num_tracks', MAX_NUM_TRACKS)
oFile.set_node_attr(ana_group_name, 'blob_radius', BLOB_RADIUS)
oFile.set_node_attr(ana_group_name, 'blob_Eth', BLOB_E_THRESHOLD)
oFile.set_node_attr(ana_group_name, 'roi_Emin', ROI_E_MIN)
oFile.set_node_attr(ana_group_name, 'roi_Emax', ROI_E_MAX)



#--- Output data

events_dict = get_events_ana_dict()
voxels_dict = get_voxels_ana_dict()



#--- Analysis procedure

# Getting the events & voxels data from the reconstruction phase
events_df = pd.read_hdf(iFileName, reco_group_name + '/events')
voxels_df = pd.read_hdf(iFileName, reco_group_name + '/voxels')

# Identifying as negligible all the voxels with energy lower than threshold
voxels_df['negli'] = voxels_df.E < VOXEL_E_THRESHOLD

if VERBOSITY_LEVEL >= 1:
    print('\n* Total Voxels in File: {0}     Negligible Voxels (below {1:3.1f} keV): {2}\n'
          .format(len(voxels_df), VOXEL_E_THRESHOLD/units.keV,
          	      len(voxels_df[voxels_df.negli == True])))
    print('* Analyzing events ...')


# Analyzing only the fiducial events ...

# Counter of analyzed events for verbosing pourpouses
num_analyzed_events = 0

for event_id in events_df[events_df.fid_filter].index:
    
    # Updating counter of analyzed events
    num_analyzed_events += 1
    
    # Verbosing
    if (VERBOSITY_LEVEL >= 1):
        if (num_analyzed_events % 100 == 0):
            print('  {} analized events ...'.format(num_analyzed_events))

    if (VERBOSITY_LEVEL >= 2):
        print('\n* Analyzing event Id: {0} ...'.format(event_id))

    # Getting the voxels of current event and their sizes
    event_voxels = voxels_df[voxels_df.event_id == event_id]
    num_event_voxels = len(event_voxels)
    num_event_voxels_negli = len(event_voxels[event_voxels.negli == True])
    event_data = events_df.loc[event_id]
    voxel_dimensions = (event_data.voxel_sizeX, event_data.voxel_sizeY, event_data.voxel_sizeZ)
    
    if (VERBOSITY_LEVEL >= 2):
        print('  Total Voxels: {}   Negli. Voxels: {}   Voxels Size: ({:3.1f}, {:3.1f}, {:3.1f}) mm'\
              .format(num_event_voxels, num_event_voxels_negli, voxel_dimensions[0],
                     voxel_dimensions[1], voxel_dimensions[2]))

    # If there is any negligible Voxel, distribute its energy between its neighbours,
    # if not, all voxels maintain their previous energies
    if num_event_voxels_negli:
        event_voxels_newE = get_new_energies(event_voxels, (VERBOSITY_LEVEL>=3))
    else:
        event_voxels_newE = event_voxels.E.tolist()
    
    # Translate fanalIC voxels info to IC voxels to make tracks
    ic_voxels = [Voxel(event_voxels.iloc[i].X, event_voxels.iloc[i].Y, event_voxels.iloc[i].Z,
                       event_voxels_newE[i], voxel_dimensions) for i in range(num_event_voxels)]
    
    # Make tracks
    event_tracks = make_track_graphs(ic_voxels)
    num_event_tracks = len(event_tracks)
    if VERBOSITY_LEVEL >= 2:
        print('  Num initial tracks: {:2}'.format(num_event_tracks))
        
    # Appending to every voxel, the track it belongs to
    event_voxels_tracks = get_voxel_track_relations(event_voxels, event_tracks)
    
    # Appending info of this event voxels
    extend_voxels_ana_data(voxels_dict, event_voxels.index, event_voxels_newE, event_voxels_tracks)

    # Processing tracks: Getting energies, sorting and filtering ...
    event_sorted_tracks = process_tracks(event_tracks, TRACK_E_THRESHOLD, (VERBOSITY_LEVEL>=3))         
    num_event_tracks = len(event_sorted_tracks)

    # Storing 3 hottest track info
    if num_event_tracks >= 1:
        track0_E      = event_sorted_tracks[0][0]
        track0_voxels = len(event_sorted_tracks[0][1].nodes())
    else:
        track0_E = track0_voxels = np.nan
    if num_event_tracks >= 2:
        track1_E      = event_sorted_tracks[1][0]
        track1_voxels = len(event_sorted_tracks[1][1].nodes())
    else:
        track1_E = track1_voxels = np.nan
    if num_event_tracks >= 3:
        track2_E      = event_sorted_tracks[2][0]
        track2_voxels = len(event_sorted_tracks[2][1].nodes())
    else:
        track2_E = track2_voxels = np.nan
    
    # Applying the tracks filter
    tracks_filter = ((num_event_tracks > 0) & (num_event_tracks <= MAX_NUM_TRACKS))

    # Verbosing
    if VERBOSITY_LEVEL >= 2:
        print('  Num final tracks:   {:2}  -->  tracks_filter: {}'.format(num_event_tracks, tracks_filter))
    
    # For those events NOT passing the tracks filter:
    # Storing data of NON tracks_filter vents
    if not tracks_filter:
        extend_events_ana_data(events_dict, event_id, num_event_tracks, track0_E, track0_voxels, track1_E,
                               track1_voxels, track2_E, track2_voxels, tracks_filter)
            
    # Only for those events passing the tracks filter:
    else:
        
        # Getting the blob energies of the track with highest energy
        blobs_E = blob_energies(event_sorted_tracks[0][1], BLOB_RADIUS)
        blob1_E = blobs_E[1]
        blob2_E = blobs_E[0]
        
        # Applying the blobs filter
        blobs_filter = (blob2_E > BLOB_E_THRESHOLD)
        
        # Verbosing
        if VERBOSITY_LEVEL >= 2:
            print('  Blob 1 energy: {:4.1f} keV   Blob 2 energy: {:4.1f} keV  -->  Blobs filter: {}'\
                  .format(blob1_E/units.keV, blob2_E/units.keV, blobs_filter))

        # For those events NOT passing the blobs filter:
        # Storing data of NON blobs_filter vents
        if not blobs_filter:
            extend_events_ana_data(events_dict, event_id, num_event_tracks, track0_E, track0_voxels, track1_E,
                                   track1_voxels, track2_E, track2_voxels, tracks_filter,
                                   blob1_E = blob1_E, blob2_E = blob2_E, blobs_filter = blobs_filter)
            
        # Only for those events passing the blobs filter:
        else:
            
            # Getting the total event smeared energy
            event_smE = events_df.loc[event_id].smE
            
            # Applying the ROI filter
            roi_filter = ((event_smE >= ROI_E_MIN) & (event_smE <= ROI_E_MAX))
        
            # Verbosing
            if VERBOSITY_LEVEL >= 2:
                print('  Event energy: {:6.1f} keV  -->  ROI filter: {}'\
                      .format(event_smE / units.keV, roi_filter))
                
            # Storing all the events (as this is the last filter)
            extend_events_ana_data(events_dict, event_id, num_event_tracks, track0_E, track0_voxels,
            					   track1_E, track1_voxels, track2_E, track2_voxels, tracks_filter,
                                   blob1_E = blob1_E, blob2_E = blob2_E, blobs_filter = blobs_filter,
								   roi_filter = roi_filter)



#--- Generating and storing the "events" and "voxels" DataFrame      

print('\n* Storing the analysis data ... \n')
store_events_ana_data(oFileName, ana_group_name, events_df, events_dict)
store_voxels_ana_data(oFileName, ana_group_name, voxels_df, voxels_dict)

# Event counters as attributes
tracks_filter_events, blobs_filter_events, roi_filter_events = \
	store_events_ana_counters(oFileName, ana_group_name, events_df)

# Closing the output file
oFile.close()

print('\n* fanalIC analysis done!\n')



#--- Printing and plotting results

## Priting the event counters
with tb.open_file(iFileName, mode='a') as iFile:
    simulated_events  = iFile.get_node_attr(reco_group_name, 'simulated_events')
    stored_events     = iFile.get_node_attr(reco_group_name, 'stored_events')
    smE_filter_events = iFile.get_node_attr(reco_group_name, 'smE_filter_events')
    fid_filter_events = iFile.get_node_attr(reco_group_name, 'fid_filter_events')
print('''* Event counters:
Simulated events:     {0:8}
Stored events:        {1:8}
smE_filter events:    {2:8}
fid_filter events:    {3:8}
tracks_filter events: {4:8}
blobs_filter events:  {5:8}
roi_filter events:    {6:8}
'''.format(simulated_events, stored_events, smE_filter_events, fid_filter_events,
		   tracks_filter_events, blobs_filter_events, roi_filter_events))


events_df = pd.read_hdf(oFileName, ana_group_name + '/events')
voxels_df = pd.read_hdf(oFileName, ana_group_name + '/voxels')

events_df_smE_True    = events_df[events_df.smE_filter    == True]
events_df_fid_True    = events_df[events_df.fid_filter    == True]
events_df_tracks_True = events_df[events_df.tracks_filter == True]
events_df_blobs_True  = events_df[events_df.blobs_filter  == True]
events_df_roi_True    = events_df[events_df.roi_filter    == True]

columns_toShow = ['smE', 'num_tracks', 'track0_E', 'track0_voxels', 'tracks_filter',
				  'blob1_E', 'blob2_E', 'blobs_filter', 'roi_filter']

print("\nEvents passing the tracks_filter ...\n", events_df_tracks_True[columns_toShow].head())
print("\nEvents passing the blobs_filter ...\n",  events_df_blobs_True[columns_toShow].head())
print("\nEvents passing the roi_filter ...\n",    events_df_roi_True[columns_toShow].head())
print("\nVoxels ...\n", voxels_df.head())


## Plotting number of tracks from fiducial events
fig = plt.figure(figsize = (8,5))
num_bins = 10

plt.hist(events_df_fid_True.num_tracks, num_bins, [0, 10])
plt.xlabel('Num Tracks', size=12)
plt.ylabel('Num. events', size=12)
plt.title('{} - Number of reconstructed tracks'.format(EVENT_TYPE))

plt.show()


## Plotting energies and number of voxels of the three hottest tracks
fig = plt.figure(figsize = (18, 9))
num_E_bins = 50
num_voxels_bins = 20
#plotting_events = events_df[events_df.fid_filter == True]

# First track plots
ax1 = fig.add_subplot(2, 3, 1)
plt.hist(events_df_fid_True.track0_E, num_E_bins, [0., 2.5])
plt.title('Track 0 Energy [MeV]', size=12)

ax2 = fig.add_subplot(2, 3, 4)
plt.hist(events_df_fid_True.track0_voxels, num_voxels_bins, [0, 40])
plt.title('Track 0 - Num Voxels', size=12)

# Second track plots
plotting_events = events_df_fid_True[events_df_fid_True.num_tracks >= 2]
ax3 = fig.add_subplot(2, 3, 2)
plt.hist(plotting_events.track1_E, num_E_bins, [0., 1.0])
plt.title('Track 1 Energy [MeV]', size=12)

ax4 = fig.add_subplot(2, 3, 5)
plt.hist(plotting_events.track1_voxels, num_voxels_bins, [0, 20])
plt.title('Track 1 - Num Voxels', size=12)

# Third track plots
plotting_events = events_df_fid_True[events_df_fid_True.num_tracks >= 3]
ax5 = fig.add_subplot(2, 3, 3)
plt.hist(plotting_events.track2_E, num_E_bins, [0., 1.0])
plt.title('Track 2 Energy [MeV]', size=12)

ax6 = fig.add_subplot(2, 3, 6)
plt.hist(plotting_events.track2_voxels, num_voxels_bins, [0, 20])
plt.title('Track 2 - Num Voxels', size=12)

plt.show()


## Energy histogram of events passing the tracks filter 
fig = plt.figure(figsize = (15,5))
num_bins = 20

ax1 = fig.add_subplot(1, 2, 1)
plt.hist(events_df_tracks_True.track0_E, num_bins, [2.4, 2.5])
plt.xlabel('Track Energy [MeV]', size=14)
plt.title('{} - Track Energy'.format(EVENT_TYPE))

ax2 = fig.add_subplot(1, 2, 2)
plt.hist(events_df_tracks_True.smE, num_bins, [2.4, 2.5])
plt.xlabel('Event Energy [MeV]', size=14)
plt.title('{} - Event Energy'.format(EVENT_TYPE))

plt.show()


## Histograms of blob energies
fig = plt.figure(figsize = (15,5))
num_bins = 30

ax1 = fig.add_subplot(1, 2, 1)
plt.hist(events_df_tracks_True.blob1_E, num_bins, [0, 1.5])
plt.xlabel('Blob1 Energy [MeV]')
plt.title('{} - Blob1 Energy [MeV]'.format(EVENT_TYPE), size=14)

ax2 = fig.add_subplot(1, 2, 2)
plt.hist(events_df_tracks_True.blob2_E, num_bins, [0, 1.5])
plt.xlabel('Blob2 Energy [MeV]')
plt.title('{} - Blob2 Energy [MeV]'.format(EVENT_TYPE), size=14)

fig = plt.figure(figsize = (7,6))
plt.hist2d(events_df_tracks_True.blob1_E, events_df_tracks_True.blob2_E, num_bins,
           [[0, 1.5], [0, 1.5]], norm=LogNorm())
plt.xlabel('Highest Blob Energy [MeV]')
plt.ylabel('Lowest Blob Energy [MeV]')
plt.title('{} - Blobs Energies [MeV]'.format(EVENT_TYPE), size=14)

plt.show()


## ROI energy
fig = plt.figure(figsize = (7,6))
num_bins = int((ROI_E_MAX - ROI_E_MIN) / units.keV)

plt.hist(events_df_roi_True.smE, num_bins, [ROI_E_MIN, ROI_E_MAX])
plt.xlabel('Event Energy [MeV]')
plt.title('{} - Event Energy'.format(EVENT_TYPE), size=14)

plt.show()

