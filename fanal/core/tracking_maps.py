# General importings
import math
import pandas as pd
import numpy  as np

from   typing import List

maps_path = '../fanal/maps/'


def get_neighbour_ids(sns_id       : int,
                      tracking_map : pd.DataFrame
                     )            -> List[int] :
    """
    It returns a list with the neighbour sensor ids
    """
    neigh_ids = []
    sensor    = tracking_map.loc[sns_id]
    if (not np.isnan(sensor.id_left)) : neigh_ids.append(int(sensor.id_left))
    if (not np.isnan(sensor.id_right)): neigh_ids.append(int(sensor.id_right))
    if (not np.isnan(sensor.id_up))   : neigh_ids.append(int(sensor.id_up))
    if (not np.isnan(sensor.id_down)) : neigh_ids.append(int(sensor.id_down))
    return neigh_ids


def fill_neighbours(tracking_map_fname : pd.DataFrame,
                    pitch_x            : float,
                    pitch_y            : float
                    )           -> None :
    """
    It fills the neighbour sensor ids of the tracking_map stored in tracking_map_fname
    based on the pitchs provided, and re-write it.
    """

    tracking_map = pd.read_csv(maps_path + tracking_map_fname)
    tracking_map.set_index('sensor_id', inplace = True)

    for sensor_id, sensor in tracking_map.iterrows():

        left_sns  = tracking_map[(tracking_map.x == math.trunc((sensor.x - pitch_x) * 100) / 100) &
                                 (tracking_map.y == sensor.y)]
        if len(left_sns):  tracking_map.loc[sensor_id, 'id_left'] = left_sns.index.values[0]
        else:              tracking_map.loc[sensor_id, 'id_left'] = np.nan

        right_sns = tracking_map[(tracking_map.x == math.trunc((sensor.x + pitch_x) * 100) / 100) &
                                 (tracking_map.y == sensor.y)]
        if len(right_sns): tracking_map.loc[sensor_id, 'id_right'] = right_sns.index.values[0]
        else:              tracking_map.loc[sensor_id, 'id_right'] = np.nan

        up_sns    = tracking_map[(tracking_map.x == sensor.x) &
                                 (tracking_map.y == math.trunc((sensor.y + pitch_y) * 100) / 100)]
        if len(up_sns):    tracking_map.loc[sensor_id, 'id_up'] = up_sns.index.values[0]
        else:              tracking_map.loc[sensor_id, 'id_up'] = np.nan

        down_sns  = tracking_map[(tracking_map.x == sensor.x) &
                                 (tracking_map.y == math.trunc((sensor.y - pitch_y) * 100) / 100)]
        if len(down_sns):  tracking_map.loc[sensor_id, 'id_down'] = down_sns.index.values[0]
        else:              tracking_map.loc[sensor_id, 'id_down'] = np.nan

    tracking_map.to_csv(maps_path + tracking_map_fname)

