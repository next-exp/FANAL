"""
Module ana_io_functions.
This module includes the functions related with input/output of data of the analysis step.

Notes
-----
    FANAL code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https
"""

import os
import tables as tb
import numpy  as np
import pandas as pd

from typing   import List, Sequence, Tuple, Dict, Any

from fanal.core.fanal_types import SpatialDef



def get_ana_group_name(fwhm        : float,
                       spatial_def : SpatialDef
                      ) -> str:
  """
  Define the ana_group:

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
  
  return f"/FANALIC/ANA_{str(fwhm).replace('.', '')}fmhm_{spatial_def.name}Def"



########## EVENTS IO FUNCTIONS ##########

def get_events_ana_dict() -> Dict[str, List[Any]]:
  """
  It returns a dictionary with a key for each field to be stored per event
  during the fanalIC analysis step.
  """
  events_dict = {
		'id':            [],
    'num_tracks':    [],
    'track0_E':      [],
    'track0_voxels': [],
    'track0_length': [],
    'track1_E':      [],
    'track1_voxels': [],
    'track1_length': [],
    'track2_E':      [],
    'track2_voxels': [],
    'track2_length': [],
    'tracks_filter': [],
    'blob1_E':       [],
    'blob2_E':       [],
    'blobs_filter':  [],
    'roi_filter':    []
	}
  return events_dict



def extend_events_ana_data(events_dict   : Dict[str, List[Any]],
                           event_id      : int,
                           num_tracks    : int,
                           track0_E      : float,
                           track0_voxels : int,
                           track0_length : float,
                           track1_E      : float,
                           track1_voxels : int,
                           track1_length : float,
                           track2_E      : float,
                           track2_voxels : int,
                           track2_length : float,
                           tracks_filter : bool,
                           blob1_E       : float = np.nan,
                           blob2_E       : float = np.nan,
                           blobs_filter  : bool  = False,
                           roi_filter    : bool  = False
                          ) -> None:
  """
  It stores all the event data from the analysis into the events_dict.
  The values not passed in the function called are set to default values
  to fill all the dictionary fields per event.
  """
  events_dict['id'].extend([event_id])
  events_dict['num_tracks'].extend([num_tracks])
  events_dict['track0_E'].extend([track0_E])
  events_dict['track0_voxels'].extend([track0_voxels])
  events_dict['track0_length'].extend([track0_length])
  events_dict['track1_E'].extend([track1_E])
  events_dict['track1_voxels'].extend([track1_voxels])
  events_dict['track1_length'].extend([track1_length])
  events_dict['track2_E'].extend([track2_E])
  events_dict['track2_voxels'].extend([track2_voxels])
  events_dict['track2_length'].extend([track2_length])
  events_dict['tracks_filter'].extend([tracks_filter])
  events_dict['blob1_E'].extend([blob1_E])
  events_dict['blob2_E'].extend([blob2_E])
  events_dict['blobs_filter'].extend([blobs_filter])
  events_dict['roi_filter'].extend([roi_filter])



def store_events_ana_data(file_name   : str,
                          group_name  : str,
                          events_df   : pd.DataFrame,
                          events_dict : Dict[str, List[Any]]
                         ) -> None:
  """
  Adds the events new data (coming in events_dict), to the pre-existing events data
  only for the corresponding events (those whose event_id's are listed in the
  incoming dict).
  Then dataFrame is stored in file_name / group_name / events.
  """
  for key in events_dict.keys():
    events_df.loc[events_dict['id'], key] = events_dict[key]
    # 'filters' have boolean values so, to be stored, nan values must be set to False
    if 'filter' in key:
      events_df[key].fillna(False, inplace = True)

  events_df.sort_index()
  events_df.to_hdf(file_name, group_name + '/events', format='table', data_columns=True)



def store_events_ana_counters(oFile      : tb.file.File,
                              group_name : str,
                              events_df  : pd.DataFrame
                             ) -> Tuple[int, int, int]:
  """
  Stores the event counters as attributes of oFile / group_name
  """
  tracks_filter_events = events_df.tracks_filter.sum()
  blobs_filter_events  = events_df.blobs_filter.sum()
  roi_filter_events    = events_df.roi_filter.sum()

  oFile.set_node_attr(group_name, 'tracks_filter_events', tracks_filter_events)
  oFile.set_node_attr(group_name, 'blobs_filter_events',  blobs_filter_events)
  oFile.set_node_attr(group_name, 'roi_filter_events',    roi_filter_events)

  return tracks_filter_events, blobs_filter_events, roi_filter_events



########## VOXELS IO FUNCTIONS ##########

def get_voxels_ana_dict() -> Dict[str, List[Any]]:
	"""
	It returns a dictionary with a key for each field to be stored per voxel
	during the fanalIC analysis step.
	"""
	voxels_dict = {
		'indexes': [],
		'newE':    [],
		'trackID': []
	}

	return voxels_dict



def extend_voxels_ana_data(voxels_dict    : Dict[str, List[Any]],
                           voxels_indexes : pd.core.indexes.numeric.Int64Index,
                           voxels_newE    : List[float],
                           voxels_trackID : List[int]
                          ) -> None:
	"""
	It stores all the data related with the analysis of voxels
	into the voxels_dict.
	"""
	voxels_dict['indexes'].extend(voxels_indexes)
	voxels_dict['newE'].extend(voxels_newE)
	voxels_dict['trackID'].extend(voxels_trackID)



def store_voxels_ana_data(file_name   : str,
                          group_name  : str,
                          voxels_df   : pd.DataFrame,
                          voxels_dict : Dict[str, List[Any]]
                         ) -> None:
	"""
  Adds the voxels new data (coming in voxels_dict), to the pre-existing voxels data
  only for the corresponding voxels (those whose voxel_id's are listed in the
  incoming dict).
  Then dataFrame is stored in	file_name / group_name / voxels.
	"""
	voxels_df.loc[voxels_dict['indexes'], 'newE'] = voxels_dict['newE']
	voxels_df.loc[voxels_dict['indexes'], 'track_id'] = voxels_dict['trackID']
	#voxels_df.to_hdf(file_name, group_name + '/voxels', format='table', data_columns='evt_id')
	voxels_df.to_hdf(file_name, group_name + '/voxels', format='table', data_columns=True)
