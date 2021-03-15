"""
Tests for logger
"""

import logging


from fanal.utils.logger         import get_logger

def test_logger():
	log_file_name = 'logger_test.log'
	logger = get_logger('Test Logger', logging.INFO, log_file_name)
	
	logger.warning('This is a warning message.')
	logger.info('This is an info message.')
	logger.debug('This is a debug message.')

	log_file = open(log_file_name, 'r')
	assert log_file.read() == 'Test Logger - This is a warning message.\n' + \
                              'Test Logger - This is an info message.\n'




