"""
Define Fanal-specific exceptions
"""

class FException(Exception):
    """ Base class for Fanal exceptions hierarchy """

class DetectorNameNotDefined(ICException):
    """ Detector name is not defined """
