"""
Tests for logger
"""

import numpy as np
import logging

from pytest        import mark
from pytest        import approx
from pytest        import raises
from flaky         import flaky
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from hypothesis					 import given, settings
from hypothesis.strategies 		 import integers
from hypothesis.strategies 		 import floats

from fanal.core.logger         import get_logger

def test_logger():
	log_file_name = 'logger_test.log'
	logger = get_logger('Test Logger', logging.INFO, log_file_name)
	
	logger.warning('This is a warning message.')
	logger.info('This is an info message.')
	logger.debug('This is a debug message.')

	log_file = open(log_file_name, 'r')
	assert log_file.read() == 'Test Logger - This is a warning message.\nTest Logger - This is an info message.\n'




