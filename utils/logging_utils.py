import logging, sys
from pathlib import Path
def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name); logger.setLevel(level)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        ch = logging.StreamHandler(sys.stdout); ch.setFormatter(fmt); ch.setLevel(level); logger.addHandler(ch)
        fh = logging.FileHandler(Path(log_dir)/f"{name}.log"); fh.setFormatter(fmt); fh.setLevel(level); logger.addHandler(fh)
    return logger
