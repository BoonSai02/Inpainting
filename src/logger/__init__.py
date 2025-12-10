import logging
import sys
import os
from datetime import datetime

# Setup logs directory
# src/logger/logs
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Generate log filename based on current time
# One log file per session (import)
LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE_NAME)

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Stream Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # File Handler
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
