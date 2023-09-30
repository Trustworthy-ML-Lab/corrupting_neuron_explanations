import logging
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)


def get_logger(name: str):
    return logging.getLogger(name)


def set_log_level(level: str):
    if level == "DEBUG":
        logging.getLogger().setLevel(logging.DEBUG)
    elif level == "INFO":
        logging.getLogger().setLevel(logging.INFO)
    elif level == "WARNING":
        logging.getLogger().setLevel(logging.WARNING)
    elif level == "ERROR":
        logging.getLogger().setLevel(logging.ERROR)
    elif level == "CRITICAL":
        logging.getLogger().setLevel(logging.CRITICAL)
    else:
        raise ValueError("Invalid logging level")


def set_log_file_handler(path: Path):
    logging.getLogger().addHandler(logging.FileHandler(path))
