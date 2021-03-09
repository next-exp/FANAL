import pytest

import invisible_cities.core.system_of_units               as units
from   invisible_cities.evm.event_model       import Voxel as icVoxel

from fanal.core.detectors        import get_detector
from fanal.core.fanal_exceptions import DetectorNameNotDefined


def test_wrong_detector():
    """Checks that NotDefined detectors raise the corresponding exception"""
    with pytest.raises(DetectorNameNotDefined):
        get_detector("NotValidName")


def test_symmetric_fiducial_checker():
    """Checks that fiducial_checker of symmetric detectors"""
    detector = get_detector("NEXT_HD")
    assert detector.symmetric

    fiduc_checker = detector.get_fiducial_checker(20 * units.mm)
    # Check fiducial voxel
    assert fiduc_checker(icVoxel(100, 100, 100, 0.003, (10., 10., 10.)))
    # Check non-fiducial voxel with negative Z
    assert not fiduc_checker(icVoxel(0, 0, -1290, 0.003, (10., 10., 10.)))
    # Check non-fiducial voxel with positive Z
    assert not fiduc_checker(icVoxel(0, 0, 1290, 0.003, (10., 10., 10.)))
    # Check non-fiducial voxel at center
    assert not fiduc_checker(icVoxel(0, 0, 0, 0.003, (10., 10., 10.)))
    # Check non-fiducial voxel with high radius
    assert not fiduc_checker(icVoxel(0, 1290, 0, 0.003, (10., 10., 10.)))


def test_asymmetric_fiducial_checker():
    """Checks that fiducial_checker of asymmetric detectors"""
    detector = get_detector("NEXT100")
    assert not detector.symmetric

    fiduc_checker = detector.get_fiducial_checker(20 * units.mm)
    # Check fiducial voxel
    assert fiduc_checker(icVoxel(100, 100, 100, 0.003, (10., 10., 10.)))
    # Check non-fiducial voxel with low Z
    assert not fiduc_checker(icVoxel(0, 0, 10, 0.003, (10., 10., 10.)))
    # Check non-fiducial voxel with high Z
    assert not fiduc_checker(icVoxel(0, 0, 1190, 0.003, (10., 10., 10.)))
    # Check non-fiducial voxel with high radius
    assert not fiduc_checker(icVoxel(0, 1190, 0, 0.003, (10., 10., 10.)))

