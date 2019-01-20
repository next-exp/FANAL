import logging
import sys

from typing import Any, Union

def get_logger(name     : str,
	           level    : int = logging.WARNING,
	           filename : str = ''):

	# Switching off the default logger
	root_logger = logging.getLogger()
	root_logger.handlers = []

	# Create the logger
	logger = logging.getLogger(name)

	# Create handlers
	if filename == '':
		screen_handler        = logging.StreamHandler()
		screen_handler.stream = sys.stdout
		screen_handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
		logger.addHandler(screen_handler)
	else:
		file_handler = logging.FileHandler(filename, mode='w')
		file_handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))
		logger.addHandler(file_handler)

	# Setting verbosity level
	logger.setLevel(level)

	return logger
