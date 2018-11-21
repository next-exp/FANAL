import os

from invisible_cities.core.system_of_units_c  import units
from fanal.core.fanal_types      import DetName
from fanal.core.fanal_types      import ActiveVolumeDim
from fanal.core.fanal_exceptions import DetectorNameNotDefined

def get_active_size(detname : DetName) -> ActiveVolumeDim:
	"""
	It returns the size of the ACTIVE region of the detector:
	z_min, z_max, rad [mm]
	"""

	if detname == DetName.next100:
		return ActiveVolumeDim(
		                       z_min =    0.  * units.mm,
		                       z_max = 1300.  * units.mm,
		                       rad   =  534.5 * units.mm)

	elif det_name == DetName.new:
		return ActiveVolumeDim(z_min =   0.  * units.mm,
						       z_max = 532.  * units.mm,
							   rad   = 198.  * units.mm)
	elif det_name == DetName.next500:
		return ActiveVolumeDim(z_min =   0.   * units.mm,
						       z_max = 2000.  * units.mm,
							   rad   = 1000.  * units.mm)
	else:
		except DetectorNameNotDefined





#########################################################################################
if __name__ == "__main__":

	DET_NAME = 'NEW'

	print(get_active_size(DET_NAME))
