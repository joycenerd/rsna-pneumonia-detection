import logging
import logging.handlers

BACKUP_COUNT = 5
LOG_FORMAT = '%(asctime)s - %(filename)s:%(lineno)s - %(name)s:%(funcName)s - [%(levelname)s] %(message)s'

def setup_logger(name, log_file, max_size, level):
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=BACKUP_COUNT
    )
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger