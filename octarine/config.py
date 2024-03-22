import logging
import os

HEADLESS = False
PRIMARY_VIEWER = None

logger = logging.getLogger('octarine')


def default_logging():
    """Add a formatted stream handler to the ``octarine`` logger.

    Called by default when octarine is imported for the first time.
    To prevent this behaviour, set an environment variable:
    ``OCTARINE_SKIP_LOG_SETUP=True``.
    """
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(levelname)-5s : %(message)s (%(name)s)')
        sh.setFormatter(formatter)
        logger.addHandler(sh)


def remove_log_handlers():
    """Remove all handlers from the ``octarine`` logger.

    It may be preferable to skip octarine's default log handler
    being added in the first place.
    Do this by setting an environment variable before the first import:
    ``OCTARINE_SKIP_LOG_SETUP=True``.
    """
    logger.handlers.clear()


skip_log_setup = os.environ.get('OCTARINE_SKIP_LOG_SETUP', '').lower() == 'true'
if not skip_log_setup:
    default_logging()


def get_logger(name: str):
    if skip_log_setup:
        return logging.getLogger(name)
    return logger