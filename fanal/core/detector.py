import os

from invisible_cities.core.system_of_units_c  import units

from fanal.core.fanal_types      import DetName
from fanal.core.fanal_types      import VolumeDim
from fanal.core.fanal_exceptions import DetectorNameNotDefined


def get_active_size(detname : DetName) -> VolumeDim:
	"""
	It returns the size of the ACTIVE region of the detector:
	z_min, z_max, rad grouped into a VolumeDim.
	"""

	if detname == DetName.next100:
		return VolumeDim(z_min =    0.  * units.mm,
			               z_max = 1300.  * units.mm,
			               rad   =  534.5 * units.mm)

	elif detname == DetName.new:
		return VolumeDim(z_min =   0. * units.mm,
			               z_max = 532. * units.mm,
			               rad   = 198. * units.mm)

	elif detname == DetName.next500:
		return VolumeDim(z_min = -1000. * units.mm,
			               z_max =  1000. * units.mm,
			               rad   =  1000. * units.mm)

	else:
		raise DetectorNameNotDefined



def get_fiducial_size(active_size: VolumeDim, veto_width: float) -> VolumeDim:
	"""
	Computes the dimensions of the fiducial volume.

  Parameters
  ----------
  active_size:
    A VolumeDim representing the dimensions of the detector ACTIVE volume.
  veto_width:
    A float with the width of the veto region.

  Returns
  -------
    A VolumeDim with the dimensions of the fiducial volume
	"""

	return VolumeDim(z_min = active_size.z_min + veto_width,
		               z_max = active_size.z_max - veto_width,
		               rad   = active_size.rad   - veto_width)

