"""
Module reco_io_functions.
This module includes the functions related with input/output data
from the reconstruction step.

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
import tables as tb

from typing                              import Dict, List, Any, Tuple
from invisible_cities.evm.event_model    import Voxel
from fanal.core.fanal_types              import SpatialDef



def get_reco_group_name(fwhm        : float,
                        spatial_def : SpatialDef
                       ) -> str:
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



############### EVENTS IO FUNCTIONS ###############

def get_event_reco_data() -> Dict[str, Any]:
    """
    It returns a dictionary with a key for each field to be stored per event
    during the fanalIC reconstruction step.
    """
    event_data : Dict[str, Any] = {
        'event_id'     : np.nan,
        'num_MCparts'  : np.nan, 
        'num_MChits'   : np.nan,  
        'mcE'          : np.nan,
        'smE'          : np.nan,
        'smE_filter'   : False,
        'num_voxels'   : np.nan,
        'voxel_sizeX'  : np.nan,
        'voxel_sizeY'  : np.nan,
        'voxel_sizeZ'  : np.nan,
        'voxels_minZ'  : np.nan,
        'voxels_maxZ'  : np.nan,
        'voxels_maxRad': np.nan,
        'veto_energy'  : np.nan,
        'fid_filter'   : False
    }
    
    return event_data



def get_events_reco_dict() -> Dict[str, List[Any]]:
    """
    It returns the dictionary to store the reconstruction data from all the events.
    """
    event_data  = get_event_reco_data()    

    events_dict : Dict[str, List[Any]] = {}    
    for key in event_data.keys():
        events_dict[key] = []
    
    return events_dict



def extend_events_reco_dict(
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



def store_events_reco_dict(file_name   : str,
                           group_name  : str,
                           events_dict : Dict[str, List[Any]]
                          ) -> None:
    """
    Translates the events dictionary to a dataFrame that is stored in
    file_name / group_name / events.
    """
    # Creating the df
    events_df = pd.DataFrame(events_dict)

    # Formatting DF
    events_df.set_index('event_id', inplace = True)
    events_df.sort_index()

    # Storing DF
    #events_df.to_hdf(file_name, group_name + '/events', format = 'table')
    events_df.to_hdf(file_name, group_name + '/events',
                     format = 'table', data_columns = True)

    print('  Total Events in File: {}'.format(len(events_df)))



def store_events_reco_counters(oFile             : tb.file.File,
                               group_name        : str,
                               simulated_events  : int,
                               stored_events     : int,
                               smE_filter_events : int,
                               fid_filter_events : int
                              ) -> None:
	"""
	Stores the event counters as attributes of oFile / group_name
	"""
	oFile.set_node_attr(group_name, 'simulated_events',  simulated_events)
	oFile.set_node_attr(group_name, 'stored_events',     stored_events)
	oFile.set_node_attr(group_name, 'smE_filter_events', smE_filter_events)
	oFile.set_node_attr(group_name, 'fid_filter_events', fid_filter_events)



############### VOXELS IO FUNCTIONS ###############

def get_voxels_reco_dict() -> Dict[str, List[Any]]:
	"""
	It returns a dictionary with a key for each field to be stored per voxel
	during the fanalIC reconstruction step.
	"""
	voxels_dict : Dict[str, List[Any]] = {
        'event_id': [],
        'voxel_id': [],
        'X'       : [],
        'Y'       : [],
        'Z'       : [],
        'E'       : [],
        'negli'   : []
    }

	return voxels_dict



def extend_voxels_reco_dict(voxels_dict : Dict[str, List[Any]],
                            event_id    : int,
                            voxel_id    : int,
                            voxel       : Voxel,
                            voxel_Eth   : float
                           ) -> None:
    """
    It stores all the data related to a voxel into the voxels_dict.
    """
    voxels_dict['event_id'].extend([event_id])
    voxels_dict['voxel_id'].extend([voxel_id])
    voxels_dict['X']       .extend([voxel.X])
    voxels_dict['Y']       .extend([voxel.Y])
    voxels_dict['Z']       .extend([voxel.Z])
    voxels_dict['E']       .extend([voxel.E])
    voxels_dict['negli']   .extend([voxel.E < voxel_Eth])



def store_voxels_reco_dict(file_name   : str,
                           group_name  : str,
                           voxels_dict : Dict[str, List[Any]]
                          ) -> None:
    """
    Translates the voxels dictionary to a dataFrame that is stored in
    file_name / group_name / voxels.
    """
    # Creating the DF
    voxels_df = pd.DataFrame(voxels_dict)

    # Formatting DF
    voxels_df.set_index(['event_id', 'voxel_id'], inplace = True)
    voxels_df.sort_index()

    # Storing DF
    #voxels_df.to_hdf(file_name, group_name + '/voxels', format='table',
    #                 data_columns='event_id')
    voxels_df.to_hdf(file_name, group_name + '/voxels',
                     format = 'table', data_columns = True)

    print('  Total Voxels in File: {}   (Negligible: {})\n'
          .format(len(voxels_df), len(voxels_df[voxels_df.negli == True])))

