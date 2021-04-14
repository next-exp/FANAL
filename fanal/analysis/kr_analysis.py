# General importings
import pandas as pd
import numpy  as np

from   typing import Tuple

# FANAL importings
from fanal.core.fanal_types    import KrAnalysisParams
from fanal.core.detectors      import Detector
from fanal.core.detectors      import S1_WIDTH
from fanal.core.sensors        import get_energy_response
from fanal.core.sensors        import get_tracking_response

from fanal.containers.kryptons import Krypton
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

    # Getting true position
    true_pos = get_kr_pos_true(mcHits)
    logger.info(f"  True pos: {true_pos}")
    krypton.set_true_pos(true_pos)
    krypton.rad_true = true_pos.rad

    # Getting true energy
    krypton.energy_true = mcHits[mcHits.label=="ACTIVE"].energy.sum()

    # Getting rec position
    #tracking_ids = detector.get_sensor_ids('TP_SiPM')
    rec_pos      = XYZ(0., 0., 0.)
    logger.info(f"  Rec. pos: {rec_pos}")
    krypton.set_rec_pos(rec_pos)

    # Getting energy
    energy_response = get_energy_response(sns_response, detector)
    krypton.s1_pes += energy_response[energy_response.time <  S1_WIDTH].charge.sum()
    krypton.s2_pes += energy_response[energy_response.time >= S1_WIDTH].charge.sum()

    # Getting tracking charge
    tracking_response = get_tracking_response(sns_response, detector)
    krypton.q_tot, krypton.q_max, hot_id = \
        get_kr_Q(tracking_response,
                 charge_th    = params.tracking_charge_th,
                 mask_att     = params.tracking_mask_att,
                 phot_det_eff = params.tracking_sns_pde)

    return krypton



def get_kr_pos_true(mcHits : pd.DataFrame) -> XYZ :
    return XYZ(mcHits.x.mean(), mcHits.y.mean(), mcHits.z.mean())



def get_kr_Q(tracking_response : pd.DataFrame,
             charge_th         : float = 0.,
             mask_att          : float = 0.,
             phot_det_eff      : float = 1.
            ) -> Tuple[float, float, int] :
    """
    Compute the charge of SiPMs after fluctuations and
    threshold, for tracking sensors in a given event
    tracking_response is an DataFrame containing the charge and time
    for each tracking sensor.
    It returns the total and maximum charge, and the sns_id with max_q
    """
    sns_charge = tracking_response.groupby('sensor_id').charge.sum()
    hot_id     = sns_charge.idxmax()

    # if the mask attenuation is more than zero we need to fluctuate
    # these numbers according to Poisson and then multiply by (1-mask_att)
    if mask_att != 0.:
        sns_charge = np.array([np.random.poisson(xi) for xi in sns_charge]) * (1. - mask_att)

    # if the PDE of the SiPM is less than one we need to fluctuate
    # these numbers according to Poisson and then multiply by the PDE
    if phot_det_eff != 1.:
        sns_charge = np.array([np.random.poisson(xi) for xi in sns_charge]) * phot_det_eff

    # if there is a threshold we apply it now.
    if charge_th > 0.:
        sns_charge = np.array([q if q >= charge_th else 0. for q in sns_charge])

    return sum(sns_charge), max(sns_charge), hot_id