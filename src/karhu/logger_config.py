# logger_config.py
import logging


def setup_logger(log_file="training.log", level=logging.INFO):
    """
    Set up the logger to log to both a file and the console.
    Args:
        log_file (str): The name of the log file.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),  # Optional: also logs to console
        ],
    )
