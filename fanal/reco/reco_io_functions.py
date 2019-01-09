"""
Module reco_io_functions.
This module includes the functions related with input/output of data of the reconstruction step.

Notes
-----
    FANAL code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https
"""

import os
import numpy  as np
import pandas as pd

from fanal.core.fanal_types import SpatialDef


def get_reco_group_name(fwhm: float, spatial_def: SpatialDef) -> str:
  """
  Define the reco_group: 

    Parameters
    ----------
    fwhm:
      A float representing FWHM at Qbb (in %).
    spatial_def:
      An SpatialDef: enum describing the level of spatial definition ('low' or 'high').

    Returns
    -------
      A string (the group name)
	"""

  return f"/FANALIC/RECO_{str(fwhm).replace('.', '')}fmhm_{spatial_def.name}Def"



########## EVENTS IO FUNCTIONS ##########

def get_events_reco_dict():
	"""
	It returns a dictionary with a key for each field to be stored per event
	during the fanalIC reconstruction step.
	The values are empty lists.
	"""
	events_dict = {
	    'id':            [],
	    'num_MCparts':   [],
	    'num_MChits':    [],
	    'mcE':           [],
	    'smE':           [],
	    'smE_filter':    [],
	    'num_voxels':    [],
	    'voxel_sizeX':   [],
	    'voxel_sizeY':   [],
	    'voxel_sizeZ':   [],
	    'voxels_minZ':   [],
	    'voxels_maxZ':   [],
	    'voxels_maxRad': [],
	    'veto_energy':   [],
	    'fid_filter':    []
	}

	return events_dict



def extend_events_reco_data(
	events_dict,
	event_id,
	evt_num_MCparts = np.nan,
	evt_num_MChits  = np.nan,
	evt_mcE = np.nan,
	evt_smE = np.nan,
	evt_smE_filter = False,
	evt_num_voxels  = np.nan,
	evt_voxel_sizeX = np.nan,
	evt_voxel_sizeY = np.nan,
	evt_voxel_sizeZ = np.nan,
	evt_voxels_minZ = np.nan,
	evt_voxels_maxZ = np.nan,
	evt_voxels_maxRad = np.nan,
	evt_veto_energy = np.nan,
	evt_fid_filter = False
	):
	"""
	It stores all the data related to an event into the events_dict.
	"""
	events_dict['id'].extend([event_id])
	events_dict['num_MCparts'].extend([evt_num_MCparts])
	events_dict['num_MChits'].extend([evt_num_MChits])
	events_dict['mcE'].extend([evt_mcE])
	events_dict['smE'].extend([evt_smE])
	events_dict['smE_filter'].extend([evt_smE_filter])
	events_dict['num_voxels'].extend([evt_num_voxels])
	events_dict['voxel_sizeX'].extend([evt_voxel_sizeX])
	events_dict['voxel_sizeY'].extend([evt_voxel_sizeY])
	events_dict['voxel_sizeZ'].extend([evt_voxel_sizeZ])
	events_dict['voxels_minZ'].extend([evt_voxels_minZ])
	events_dict['voxels_maxZ'].extend([evt_voxels_maxZ])
	events_dict['voxels_maxRad'].extend([evt_voxels_maxRad])
	events_dict['veto_energy'].extend([evt_veto_energy])
	events_dict['fid_filter'].extend([evt_fid_filter])

# Alternative implementation (more elegant but may be slightly slower)
#def extend_events_reco_data(events_dict, event_id, **kwargs):
#	events_dict['id']           .extend([event_id])
#	events_dict['num_MCparts']  .extend([kwargs.get('evt_num_MCparts',   np.nan)])
#	events_dict['num_MChits']   .extend([kwargs.get('evt_num_MChits',    np.nan)])
#	events_dict['mcE']          .extend([kwargs.get('evt_mcE',           np.nan)])
#	events_dict['smE']          .extend([kwargs.get('evt_smE',           np.nan)])
#	events_dict['smE_filter']   .extend([kwargs.get('evt_smE_filter',    False)])
#	events_dict['num_voxels']   .extend([kwargs.get('evt_num_voxels',    np.nan)])
#	events_dict['voxel_sizeX']  .extend([kwargs.get('evt_voxel_sizeX',   np.nan)])
#	events_dict['voxel_sizeY']  .extend([kwargs.get('evt_voxel_sizeY',   np.nan)])
#	events_dict['voxel_sizeZ']  .extend([kwargs.get('evt_voxel_sizeZ',   np.nan)])
#	events_dict['voxels_minZ']  .extend([kwargs.get('evt_voxels_minZ',   np.nan)])
#	events_dict['voxels_maxZ']  .extend([kwargs.get('evt_voxels_maxZ',   np.nan)])
#	events_dict['voxels_maxRad'].extend([kwargs.get('evt_voxels_maxRad', np.nan)])
#	events_dict['veto_energy']  .extend([kwargs.get('evt_veto_energy',   np.nan)])
#	events_dict['fid_filter']   .extend([kwargs.get('evt_fid_filter',    False)])



def store_events_reco_data(file_name, group_name, events_dict):
	"""
	Translates the events dictionary to a dataFrame that is stored in
	file_name / group_name / events.
	"""
	# Creating the df
	events_df = pd.DataFrame(events_dict, index = events_dict['id'])

	# Formatting DF
	events_df.sort_index()
	
	# Storing DF
	#events_df.to_hdf(file_name, group_name + '/events', format='table')
	events_df.to_hdf(file_name, group_name + '/events', format='table', data_columns=True)



def store_events_reco_counters(file, group_name, simulated_events, stored_events, 
							   smE_filter_events, fid_filter_events):
	"""
	Stores the event counters as attributes of file / group_name
	"""
	file.set_node_attr(group_name, 'simulated_events',  simulated_events)
	file.set_node_attr(group_name, 'stored_events',     stored_events)
	file.set_node_attr(group_name, 'smE_filter_events', smE_filter_events)
	file.set_node_attr(group_name, 'fid_filter_events', fid_filter_events)



########## VOXELS IO FUNCTIONS ##########

def get_voxels_reco_dict():
	"""
	It returns a dictionary with a key for each field to be stored per voxel
	during the fanalIC reconstruction step.
	The values are empty lists.
	"""
	voxels_dict = {
    	'event_id': [],
    	'X':        [],
    	'Y':        [],
    	'Z':        [],
    	'E':        []
	}

	return voxels_dict



def extend_voxels_reco_data(voxels_dict, event_id, voxel):
	"""
	It stores all the data related to a voxel into the voxels_dict.
	"""
	voxels_dict['event_id'].extend([event_id])
	voxels_dict['X'].extend([voxel.X])
	voxels_dict['Y'].extend([voxel.Y])
	voxels_dict['Z'].extend([voxel.Z])
	voxels_dict['E'].extend([voxel.E])



def store_voxels_reco_data(file_name, group_name, voxels_dict):
	"""
	Translates the voxels dictionary to a dataFrame that is stored in
	file_name / group_name / voxels.
	"""
	# Creating the df
	voxels_df = pd.DataFrame(voxels_dict)

	# Storing DF
	#voxels_df.to_hdf(file_name, group_name + '/voxels', format='table', data_columns='event_id')
	voxels_df.to_hdf(file_name, group_name + '/voxels', format='table', data_columns=True)
