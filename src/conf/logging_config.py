import logging
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_file: str | None = None) -> None:
    """Configure le logging pour l'application.

    Args:
        log_level: logging level (e.g., logging.DEBUG)
        log_file: optional path to a log file
    """
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)

    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        separator = "="*60
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{separator}\n")
            f.write(f" NOUVELLE SESSION : {now}\n")
            f.write(f"{separator}\n\n")
