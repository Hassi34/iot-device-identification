# import logging
# import logging.handlers
from loguru import logger

# def get_logger(logsFilePath: str, maxBytes: int=10240000000000, backupCount: int=0) -> object:
#     logger = logging.getLogger()
#     fh = logging.handlers.RotatingFileHandler(logsFilePath, maxBytes=maxBytes, backupCount=backupCount)
#     fh.setLevel(logging.DEBUG)#no matter what level I set here
#     formatter = logging.Formatter("[%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(lineno)s] : %(message)s")
#     fh.setFormatter(formatter)
#     logger.addHandler(fh)
#     logger.setLevel(logging.INFO)
#     return logger


def get_logger(logs_filepath: str):
    logger.add(
        logs_filepath,
        format="{time} | {level} | {name}.{module}:{line} | {message}",
        level="DEBUG",
        rotation="10 KB",
        retention="10 days",
        compression="zip",
        colorize=True,
        enqueue=True,
        catch=True,
        encoding="utf-8",
    )
    return logger
