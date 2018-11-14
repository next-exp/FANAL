import os

from invisible_cities.core.system_of_units_c  import units


def get_active_size(det_name):
	"""
	It returns the size of the ACTIVE region of the detector:
	z_min, z_max, rad [mm]
	"""

	assert det_name in ['NEXT100', 'NEW'], '{} is not a valid detector.' \
		.format(spatial_def)

	if det_name == 'NEXT100':
		z_min =    0.  * units.mm
		z_max = 1300.  * units.mm
		rad   =  534.5 * units.mm

	elif det_name == 'NEW':
		z_min =   0.  * units.mm
		z_max = 532.  * units.mm
		rad   = 198.  * units.mm

	return z_min, z_max, rad



#########################################################################################
if __name__ == "__main__":

	DET_NAME = 'NEW'

	print(get_active_size(DET_NAME))

