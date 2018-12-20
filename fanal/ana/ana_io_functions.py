import os
import tables as tb
import numpy as np
import pandas as pd



#def get_ana_file_name(path, evt_type):
#	"""
#	It returns the name of the analysis file in the 'path'.
#	"""
#	try:
#		os.stat(path)
#	except:
#		os.mkdir(path)
#
#	fileName = evt_type + '.ana.h5'
#	fileName = os.path.join(path, fileName)
#
#	return fileName



def get_ana_group_name(fwhm, spatial_def):
	"""
	It returns the name of the ana_group where analysis-data is stored.
	fwhm: FWHM at Qbb in %
	spatial_def: Spatial definition. ('Std' or 'High')
	"""
	assert spatial_def in ['Std', 'High'], '{} is not a valid Spatial Resolution' \
		.format(spatial_def)

	ana_group_name = '/FANALIC/ANA_{0}fmhm_{1}Def' \
		.format(str(fwhm).replace('.', ''), spatial_def)

	return ana_group_name



def get_events_ana_dict():
	"""
	It returns a dictionary with a key for each field to be stored per event
	during the fanalIC analysis step.
	The initial values are empty lists.
	"""
	events_dict = {
		'id':            [],
	    'numTracks':     [],
	    'track0_E':      [],
	    'track0_voxels': [],
	    'track1_E':      [],
	    'track1_voxels': [],
	    'track2_E':      [],
	    'track2_voxels': [],
	    'tracks_filter': [],
	    'blob1_E':       [],
	    'blob2_E':       [],
	    'blobs_filter':  [],
	    'roi_filter':    []
	}

	return events_dict



def get_voxels_ana_dict():
	"""
	It returns a dictionary with a key for each field to be stored per voxel
	during the fanalIC analysis step.
	The values are empty lists.
	"""
	voxels_dict = {
		'indexes': [],
		'newE':    [],
		'trackID': []
	}

	return voxels_dict



def extend_events_ana_data(
	events_dict,
	event_id,
	numTracks,
	track0_E,
	track0_voxels,
	track1_E,
	track1_voxels,
	track2_E,
	track2_voxels,
	tracks_filter,
	blob1_E       = np.nan,
	blob2_E       = np.nan,
	blobs_filter  = False,
	roi_filter    = False
	):
	"""
	It stores all the event data from the analysis into the events_dict.
	"""
	events_dict['id'].extend([event_id])
	events_dict['numTracks'].extend([numTracks])
	events_dict['track0_E'].extend([track0_E])
	events_dict['track0_voxels'].extend([track0_voxels])
	events_dict['track1_E'].extend([track1_E])
	events_dict['track1_voxels'].extend([track1_voxels])
	events_dict['track2_E'].extend([track2_E])
	events_dict['track2_voxels'].extend([track2_voxels])
	events_dict['tracks_filter'].extend([tracks_filter])
	events_dict['blob1_E'].extend([blob1_E])
	events_dict['blob2_E'].extend([blob2_E])
	events_dict['blobs_filter'].extend([blobs_filter])
	events_dict['roi_filter'].extend([roi_filter])



def extend_voxels_ana_data(voxels_dict, voxels_indexes, voxels_newE, voxels_trackID):
	"""
	It stores all the data related with the analysis of voxels
	into the voxels_dict.
	"""
	voxels_dict['indexes'].extend(voxels_indexes)
	voxels_dict['newE'].extend(voxels_newE)
	voxels_dict['trackID'].extend(voxels_trackID)



def store_events_ana_data(file_name, group_name, events_df, events_dict):
	"""
	Adds the events dictionary data, corresponding to the event_id's passed
	to the dataFrame. Then dataFrame is stored in
	file_name / group_name / events.
	"""
	events_df.loc[events_dict['id'], 'num_tracks']    = events_dict['numTracks']
	events_df.loc[events_dict['id'], 'track0_E']      = events_dict['track0_E']
	events_df.loc[events_dict['id'], 'track0_voxels'] = events_dict['track0_voxels']
	events_df.loc[events_dict['id'], 'track1_E']      = events_dict['track1_E']
	events_df.loc[events_dict['id'], 'track1_voxels'] = events_dict['track1_voxels']
	events_df.loc[events_dict['id'], 'track2_E']      = events_dict['track2_E']
	events_df.loc[events_dict['id'], 'track2_voxels'] = events_dict['track2_voxels']
	events_df.loc[events_dict['id'], 'tracks_filter'] = events_dict['tracks_filter']
	events_df.tracks_filter.fillna(False, inplace = True)
	events_df.loc[events_dict['id'], 'blob1_E']       = events_dict['blob1_E']
	events_df.loc[events_dict['id'], 'blob2_E']       = events_dict['blob2_E']
	events_df.loc[events_dict['id'], 'blobs_filter']  = events_dict['blobs_filter']
	events_df.blobs_filter.fillna(False, inplace = True)
	events_df.loc[events_dict['id'], 'roi_filter']    = events_dict['roi_filter']
	events_df.roi_filter.fillna(False, inplace = True)

	events_df.sort_index()
	events_df.to_hdf(file_name, group_name + '/events', format='table', data_columns=True)



def store_voxels_ana_data(file_name, group_name, voxels_df, voxels_dict):
	"""
	Adds the voxels dictionary data, corresponding to the voxel indexes passed
	to the dataFrame. Then dataFrame is stored in
	file_name / group_name / voxels.
	"""
	voxels_df.loc[voxels_dict['indexes'], 'newE'] = voxels_dict['newE']
	voxels_df.loc[voxels_dict['indexes'], 'track_id'] = voxels_dict['trackID']
	#voxels_df.to_hdf(file_name, group_name + '/voxels', format='table', data_columns='evt_id')
	voxels_df.to_hdf(file_name, group_name + '/voxels', format='table', data_columns=True)



def store_events_ana_counters(file_name, group_name, events_df):
	"""
	Stores the event counters as attributes of file / group_name
	"""
	tracks_filter_events = events_df.tracks_filter.sum()
	blobs_filter_events  = events_df.blobs_filter.sum()
	roi_filter_events    = events_df.roi_filter.sum()

	with tb.open_file(file_name, mode='a') as oFile:
		oFile.set_node_attr(group_name, 'tracks_filter_events', tracks_filter_events)
		oFile.set_node_attr(group_name, 'blobs_filter_events',  blobs_filter_events)
		oFile.set_node_attr(group_name, 'roi_filter_events',    roi_filter_events)

	return tracks_filter_events, blobs_filter_events, roi_filter_events



#########################################################################################
if __name__ == "__main__":

	FWHM = 0.7
	DEF  = 'Std'
	SIM_PATH  = '/Users/Javi/Development/fanalIC_NB/data/sim'
	RECO_PATH = '/Users/Javi/Development/fanalIC_NB/data/reco'
	ANA_PATH = '/Users/Javi/Development/fanalIC_NB/data/ana'
	EVENT_TYPE = 'bb0nu'

