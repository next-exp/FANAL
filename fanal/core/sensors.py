import pandas as pd

from invisible_cities.core.core_functions  import in_range

from fanal.core.detectors                  import Detector



def get_sensor_response(sns_response : pd.DataFrame,
                        detector     : Detector,
                        sns_type     : str
                       ) -> pd.DataFrame:
    """
    Returns a data frame with the charge and time of each sensor
    of the provided sensor type.
    """
    assert sns_type in detector.get_sensor_types(), f"{sns_type} not present in {detector.name}"

    sns_range = detector.get_sensor_ids(sns_type)
    return sns_response[in_range(sns_response.index.get_level_values("sensor_id"),
                        sns_range[0], sns_range[1])].sort_index()



def get_energy_response(sns_response : pd.DataFrame,
                        detector     : Detector
                       ) -> pd.DataFrame :
    sel = [False] * len(sns_response)
    for sns_type in detector.energy_sensors:
        sns_range = detector.get_sensor_ids(sns_type)
        sel = sel | in_range(sns_response.index.get_level_values("sensor_id"),
                             sns_range[0], sns_range[1])
    return sns_response[sel]



def get_tracking_response(sns_response : pd.DataFrame,
                          detector     : Detector
                         ) -> pd.DataFrame :
    sel = [False] * len(sns_response)
    for sns_type in detector.tracking_sensors:
        sns_range = detector.get_sensor_ids(sns_type)
        sel = sel | in_range(sns_response.index.get_level_values("sensor_id"),
                             sns_range[0], sns_range[1])
    return sns_response[sel]
