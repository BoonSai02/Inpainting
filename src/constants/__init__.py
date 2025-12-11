import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')

# SAM
SAM_CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, SAM_CHECKPOINT_NAME)
SAM_MODEL_TYPE = "vit_h"
SAM_DOWNLOAD_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# ObjectClear
OBJECT_CLEAR_REPO_ID = "jixin0101/ObjectClear"
OBJECT_CLEAR_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, "ObjectClear")

# Inference Defaults
DEFAULT_IMAGE_SIZE = 512
INFERENCE_STEPS = 20
GUIDANCE_SCALE = 2.5

# Device Configuration
USE_GPU = True

# Validation Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Response Codes
class ResponseCode:
    SUCCESS_OK = "200"
    ERR_INVALID_INPUT = "400"
    ERR_PROCESSING_FAILED = "500"
    ERR_MODEL_NOT_READY = "503"
    ERR_UNAUTHORIZED = "401"

# Resource Management
MODEL_TTL_SECONDS = 300
