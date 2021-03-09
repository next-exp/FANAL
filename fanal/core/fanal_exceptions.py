"""
Define Fanal-specific exceptions
"""

class FanalException(Exception):
    """ Base class for Fanal exceptions hierarchy """

class DetectorNameNotDefined(FanalException):
    """ Detector name is not defined """
