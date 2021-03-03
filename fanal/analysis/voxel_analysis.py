# General importings
from typing import Tuple
from typing import List
from  typing import Callable

# IC importings
from invisible_cities.evm.event_model  import Voxel  as icVoxel

# FANAL importings
from fanal.utils.logger  import get_logger

# The logger
logger = get_logger('Fanal')



def clean_voxels(event_voxels : List[icVoxel],
                 voxel_Eth    : float
                ) -> List[icVoxel]:
    """
    Delete voxels with energy lower than threshold
    and redistributes their energies.
    As a first approach, we give the energy of negligible voxels to
    the closest non-negligible voxel.
    Parameters
    ----------
    event_voxels : List[icVoxel]
        List of all the icVoxels computed by PAOLINA
    voxel_Eth    : float
        Voxel energy threshold
    Returns
    -------
    List[icVoxel] containing the updated list of icVoxels
    """
    # TODO
    return event_voxels



def check_event_fiduciality(fiducial_checker : Callable,
                            event_voxels     : List[icVoxel],
                            veto_Eth         : float
                           ) -> Tuple[float, bool]:
    """
    Checks if an event is fiducial or not.

    Parameters:
    -----------
    iducial_checker : Callable
        Function to check icVoxel fiduciality
    event_voxels : List[icVoxel]
        List of icVoxels of the event.
    veto_Eth     : float
        Veto energy threshold.

    Returns:
    --------
    Tuple of 2 values containing:
        A float with the energy deposited in veto
        A bool saying if the event is fiducial or not.
    """

    veto_energy = sum(voxel.E for voxel in event_voxels \
                      if not fiducial_checker(voxel))

    fiducial_filter = veto_energy < veto_Eth

    return veto_energy, fiducial_filter
