import logging
import sys

from typing import Any

def get_logger(name     : str,
	           level    : int = logging.WARNING,
	           filename : str = False):

	# Create the logger
	logger = logging.getLogger(name)

	# Create handlers
	if filename:
		handler = logging.FileHandler(filename, mode='w')
	else:
		handler = logging.StreamHandler()
		handler.stream = sys.stdout

	# Setting format
	handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))

	# Add handler to the logger
	logger.addHandler(handler)

	# Setting verbosity level
	logger.setLevel(level)

	return logger
