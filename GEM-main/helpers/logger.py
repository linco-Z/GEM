# logger.py
import logging
import sys
import os
from datetime import datetime

logger = logging.getLogger("MyExperiment")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter_console = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter_console)
    logger.addHandler(ch)

def setup_file_handler(run_name=None, fold_idx=None, log_dir="exp_logs"):
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_parts = [run_name if run_name else "experiment_run", timestamp]
    if fold_idx is not None:
        log_file_parts.insert(1, f"fold{fold_idx}")
    log_filename = "_".join(filter(None, log_file_parts)) + ".log"
    log_file_path = os.path.join(log_dir, log_filename)

    fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter_file = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    logger.info(f"Logs will also be output to file: {log_file_path}")