import logging
import sys
import os
from datetime import datetime
from src.constants import USE_GPU

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log file name based on timestamp
LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def get_logger(name: str):
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times if they exist (though basicConfig handles root)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Console Handler
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # File Handler (Explicitly adding to ensure it's on this specific logger if basicConfig doesn't catch it for some reason, 
        # basically standardizing)
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log usage on first init of this logger
        logger.info(f"Logger initialized. USE_GPU Configuration: {USE_GPU}")

    return logger
