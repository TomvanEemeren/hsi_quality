import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = ROOT_DIR / "logs"

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if log_file:
        LOGS_DIR.mkdir(exist_ok=True)
        fh = logging.FileHandler(LOGS_DIR / log_file)
        fh.setLevel(level)
    else:
        fh = None

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if fh:
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger