import math
import numpy as np

from typing import List

from invisible_cities.evm.event_model  import MCHit


def smear_evt_energy(mcE       : float,
	                   sigma_Qbb : float,
	                   Qbb       : float
	                  ) -> float:
	"""
	It smears and returns the montecarlo energy of the event (mcE) according to:
	the sigma at Qbb (sigma_Qbb) in absolute values (keV).

	"""
	sigma_E = sigma_Qbb * math.sqrt(mcE/Qbb)
	smE = np.random.normal(mcE, sigma_E)
	return smE


def smear_hit_energies(mcHits      : List[MCHit],
	                     conv_factor : float
	                    ) -> np.array(float):
	'''
	It smears the montecarlo hit energies according to the conv_factor
	It returns a list of the hit smeared energies.
	'''
	mcEnergies = np.array([hit.E for hit in mcHits])
	smEnergies = mcEnergies * conv_factor
	return smEnergies
