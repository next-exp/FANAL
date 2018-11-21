import math
import numpy as np


def smear_evt_energy(mcE, sigma_Qbb, Qbb):
	"""
	It smears the montecarlo energy of the event (mcE) according to:
	SIGMA_Qbb the sigma at Qbb (in absolute values)
	and the Qbb.
	"""
	sigma_E = sigma_Qbb * math.sqrt(mcE/Qbb)
	smE = np.random.normal(mcE, sigma_E)
	return smE


def smear_hit_energies(mcHits, conv_factor):
	'''
	It smears the montecarlo hit energies according to the conv_factor
	It returns a list of the hit smeared energies.
	'''
	mcEnergies = np.array([hit.E for hit in mcHits])
	smEnergies = mcEnergies * conv_factor
	return smEnergies
