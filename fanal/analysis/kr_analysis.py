# General importings
import pandas as pd
import numpy  as np

from   typing import Tuple

# FANAL importings
from fanal.core.fanal_types    import KrAnalysisParams
from fanal.core.detectors      import Detector
from fanal.core.detectors      import S1_WIDTH
from fanal.core.detectors      import DRIFT_VELOCITY
from fanal.core.sensors        import get_energy_response
from fanal.core.sensors        import get_tracking_response
from fanal.core.tracking_maps  import get_sensor_pos

from fanal.containers.kryptons import Krypton

from fanal.utils.general_utils import get_barycenter
from fanal.utils.types         import XYZ
from fanal.utils.logger        import get_logger


# The logger
logger = get_logger('Fanal')



def analyze_kr_event(detector     : Detector,
                     params       : KrAnalysisParams,
                     event_id     : int,
                     sns_response : pd.DataFrame,
                     mcHits       : pd.DataFrame
                    )            -> Krypton :

    krypton = Krypton()
    krypton.krypton_id = event_id

    ### True stuff
    # Getting true energy
    krypton.energy_true = mcHits[mcHits.label=="ACTIVE"].energy.sum()

    # Getting true position
    pos_true = get_kr_pos_true(mcHits)
    logger.info(f"  True pos: {pos_true}")
    krypton.set_pos_true(pos_true)
    krypton.rad_true = pos_true.rad

    ### Reconstructed stuff
    energy_response    = get_energy_response(sns_response, detector)
    energy_s1_response = energy_response[energy_response.time <  S1_WIDTH]
    energy_s2_response = energy_response[energy_response.time >= S1_WIDTH]
    tracking_response  = get_tracking_response(sns_response, detector)
    tracking_Q         = get_tracking_intQ(tracking_response,
                                           charge_th    = params.tracking_charge_th,
                                           mask_att     = params.tracking_mask_att,
                                           phot_det_eff = params.tracking_sns_pde)

    # Storing energy pes
    krypton.s1_pes = energy_s1_response.charge.sum()
    krypton.s2_pes = energy_s2_response.charge.sum()

    # Getting rec position
    pos_rec = get_kr_pos_rec(tracking_Q,
                             energy_s2_response,
                             detector.tracking_map)
    logger.info(f"  Rec. pos: {pos_rec}")
    krypton.set_pos_rec(pos_rec)
    krypton.rad_rec = pos_rec.rad

    # Getting tracking charge
    krypton.q_tot = tracking_Q.sum()
    krypton.q_max = tracking_Q.max()

    return krypton



def get_kr_pos_true(mcHits : pd.DataFrame) -> XYZ :
    """
    True position is the mean pos of all the hits
    """
    return XYZ(mcHits.x.mean(), mcHits.y.mean(), mcHits.z.mean())




def get_pos_q(sensor_id    : int,
              tracking_map : pd.DataFrame,
              tracking_Q   : pd.Series
             )            -> Tuple[XYZ, float]:

    pos = get_sensor_pos(sensor_id, tracking_map)

    try:             q = tracking_Q.loc[sensor_id]
    except KeyError: q = 0.

    return pos, q



def get_kr_pos_rec(tracking_Q   : pd.Series,
                   energy_s2    : pd.DataFrame,
                   tracking_map : pd.DataFrame
                  )            -> XYZ :
    """
    Reconstruct XY position from barycenter around sensor with max charge.
    Reconstruct  Z position from s2 starting time * drift_velocity
    """
    hot_id  = tracking_Q.idxmax()
    hot_sns = tracking_map.loc[hot_id]
    hot_q   = tracking_Q.loc[hot_id]

    if np.isnan(hot_sns.id_left): left_pos, left_q = XYZ(0., 0., 0.), 0.
    else: left_pos, left_q = get_pos_q(hot_sns.id_left, tracking_map, tracking_Q)

    if np.isnan(hot_sns.id_right): right_pos, right_q = XYZ(0., 0., 0.), 0.
    else: right_pos, right_q = get_pos_q(hot_sns.id_right, tracking_map, tracking_Q)

    if np.isnan(hot_sns.id_up): up_pos, up_q = XYZ(0., 0., 0.), 0.
    else: up_pos, up_q = get_pos_q(hot_sns.id_up, tracking_map, tracking_Q)

    if np.isnan(hot_sns.id_down): down_pos, down_q = XYZ(0., 0., 0.), 0.
    else: down_pos, down_q = get_pos_q(hot_sns.id_down, tracking_map, tracking_Q)

    # Reconstructing position
    x_rec = get_barycenter(np.array([hot_sns.x, left_pos.x, right_pos.x]),
                           np.array([hot_q    , left_q    , right_q]))

    y_rec = get_barycenter(np.array([hot_sns.y, up_pos.y, down_pos.y]),
                           np.array([hot_q    , up_q    , down_q]))

    z_rec = energy_s2.time.min() * DRIFT_VELOCITY

    return XYZ(x_rec, y_rec, z_rec)



def get_tracking_intQ(tracking_response : pd.DataFrame,
                      charge_th         : float = 0.,
                      mask_att          : float = 0.,
                      phot_det_eff      : float = 1.
                     )                 -> pd.Series :
    """
    Computes the integrated charge for all time bins
    of tracking sensors after fluctuations and threshold in a given event
    tracking_response is an DataFrame containing the charge and time
    for each tracking sensor.
    """
    sns_charge = tracking_response.groupby('sensor_id').charge.sum().to_frame()
    sns_charge['q'] = sns_charge['charge']

    # if the mask attenuation is more than zero, we need to recalculate the mean by
    # multiplyig by (1-mask_att) and then fluctuate according to Poisson
    if mask_att != 0.:
        sns_charge.q = np.random.poisson(sns_charge.q * (1. - mask_att))

    # if the PDE of the SiPM is less than one we need to recalculate the mean by
    # multiplyig by the PDE and then fluctuate according to Poisson
    if phot_det_eff != 1.:
        sns_charge.q = np.random.poisson(sns_charge.q * phot_det_eff)

    # if there is a threshold we apply it now.
    if charge_th > 0.:
        sns_charge.q.values[sns_charge.q.values < charge_th] = 0.

    #return sns_charge
    return sns_charge.q
