import logging
import sys
import unittest

from pynif3d import logger

if __name__ == "__main__":

    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    testsuite = unittest.TestLoader().discover(".")
    success = unittest.TextTestRunner(verbosity=1).run(testsuite).wasSuccessful()

    if not success:
        sys.exit(1)
