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



############### EVENTS IO FUNCTIONS ###############

def get_event_ana_data() -> Dict[str, Any]:
    """
    It returns a dictionary with a key for each field to be stored per event
    during the fanalIC analysis step.
    """
    event_data : Dict[str, Any] = {
        'event_id'     : np.nan,
        'num_tracks'   : np.nan,
        'track0_E'     : np.nan,
        'track0_voxels': np.nan,
        'track0_length': np.nan,
        'track1_E'     : np.nan,
        'track1_voxels': np.nan,
        'track1_length': np.nan,
        'track2_E'     : np.nan,
        'track2_voxels': np.nan,
        'track2_length': np.nan,
        'tracks_filter': False,
        'blob1_E'      : np.nan,
        'blob2_E'      : np.nan,
        'blobs_filter' : False,
        'roi_filter'   : False
    }
    
    return event_data



def get_events_ana_dict() -> Dict[str, List[Any]]:
    """
    It returns the dictionary to store the analysis data from all the events.
    """
    
    event_data  = get_event_ana_data()    

    events_dict : Dict[str, List[Any]] = {}    
    for key in event_data.keys():
        events_dict[key] = []
    
    return events_dict



def extend_events_ana_dict(    
    events_dict : Dict[str, List[Any]],
    event_data  : Dict[str, Any]) -> None:
    """
    It stores all the data related to an event into the events_dict.
    The values not passed in the function called are set to default values
    to fill all the dictionary fields per event.
    """
    
    assert type(event_data['event_id']) == int, "event_id is mandatory"
    
    for key in event_data.keys():
        events_dict[key].extend([event_data[key]])



def store_events_ana_dict(file_name   : str,
                          group_name  : str,
                          events_df   : pd.DataFrame,
                          events_dict : Dict[str, List[Any]]
                         ) -> None:
    """
    Adds the events new data (coming in events_dict), to the pre-existing data
    only for the corresponding events (those whose event_id's are listed in the
    incoming dict).
    Then dataFrame is stored in file_name / group_name / events.
    """

    # Creating a new DF with the new data
    new_data_df = pd.DataFrame(events_dict)

    # Formatting the new DF        
    new_data_df.set_index('event_id', inplace=True)

    # Merging reco data with new ana data
    events_df = events_df.merge(new_data_df, left_index=True,
                                right_index=True, how='left')

    # Formatting the resulting DF
    # Boolean and np.nan can´t be mixed in the same column
    for key in events_dict.keys():
        if 'filter' in key:
            events_df[key].fillna(False, inplace = True)
    events_df.sort_index()


    # Storing the DF
    events_df.to_hdf(file_name, group_name + '/events',
                     format='table', data_columns=True)

    print('  Total Events in File: {}'.format(len(events_df)))




def store_events_ana_counters(oFile                : tb.file.File,
                              group_name           : str,
                              simulated_events     : int,
                              stored_events        : int,
                              smE_filter_events    : int,
                              fid_filter_events    : int,
                              tracks_filter_events : int,
                              blobs_filter_events  : int,
                              roi_filter_events    : int
                             ) -> None:
    """
    Stores the event counters as attributes of oFile / group_name
    """
    oFile.set_node_attr(group_name, 'simulated_events',     simulated_events)
    oFile.set_node_attr(group_name, 'stored_events',        stored_events)
    oFile.set_node_attr(group_name, 'smE_filter_events',    smE_filter_events)
    oFile.set_node_attr(group_name, 'fid_filter_events',    fid_filter_events)
    oFile.set_node_attr(group_name, 'tracks_filter_events', tracks_filter_events)
    oFile.set_node_attr(group_name, 'blobs_filter_events',  blobs_filter_events)
    oFile.set_node_attr(group_name, 'roi_filter_events',    roi_filter_events)



############### VOXELS IO FUNCTIONS ###############

def get_voxels_ana_dict() -> Dict[str, List[Any]]:
	"""
	It returns a dictionary with a key for each field to be stored per voxel
	during the fanalIC analysis step.
	"""
	voxels_dict : Dict[str, List[Any]] = {
        'event_id' : [],
        'voxel_id' : [],
		'newE'     : [],
		'track_id' : []
	}

	return voxels_dict



def extend_voxels_ana_dict(voxels_dict     : Dict[str, List[Any]],
                           event_id        : int,
                           voxel_id        : List[int],
                           voxels_newE     : List[float],
                           voxels_track_id : List[int]
                          ) -> None:
    """
    It stores all the data related with the analysis of voxels
    into the voxels_dict.
    """

    # Checking all Lists have the same length
    assert (len(voxel_id) == len(voxels_newE) == len(voxels_track_id)), \
        "extend_voxels_ana_data. All the lists must have the same length. {} {} {}" \
        .format(len(voxel_id), len(voxels_newE), len(voxels_track_id))

    voxels_dict['event_id'].extend([event_id] * len(voxel_id))
    voxels_dict['voxel_id'].extend(voxel_id)
    voxels_dict['newE']    .extend(voxels_newE)
    voxels_dict['track_id'].extend(voxels_track_id)



def store_voxels_ana_dict(file_name   : str,
                          group_name  : str,
                          voxels_df   : pd.DataFrame,
                          voxels_dict : Dict[str, List[Any]]
                         ) -> None:
    """
    Adds the voxels new data (coming in voxels_dict), to the pre-existing data
    only for the corresponding voxels (those whose voxel_id's are listed in the
    incoming dict).
    Then dataFrame is stored in	file_name / group_name / voxels.
    """

    # Creating a new DF with the new data
    new_data_df = pd.DataFrame(voxels_dict)

    # Formatting the new DF        
    new_data_df.set_index(['event_id', 'voxel_id'], inplace=True)

    # Merging reco data with new ana data
    voxels_df = voxels_df.merge(new_data_df, left_index=True,
                                right_index=True, how='left')

    # Storing the DF
    #voxels_df.to_hdf(file_name, group_name + '/voxels',
    #                 format='table', data_columns='evt_id')
    voxels_df.to_hdf(file_name, group_name + '/voxels',
                     format='table', data_columns=True)

    print('  Total Voxels in File: {}\n'.format(len(voxels_df)))

