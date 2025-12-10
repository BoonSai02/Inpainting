import torch
from PIL import Image
import numpy as np
from src.components.sam_component import SAMComponent
from src.components.object_clear_component import ObjectClearComponent
from src.logger import get_logger
from src.exceptions import CustomException

logger = get_logger(__name__)

class InpaintingPipeline:
    def __init__(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Initializing pipeline on {device}")
        
        self.sam = SAMComponent(device=str(device))
        self.object_clear = ObjectClearComponent(device=str(device))
        
    def process_request(self, image: Image.Image, points: list[list[int]]) -> Image.Image:
        """
        End-to-end processing: Image + Points -> Mask -> Inpainted Image
        """
        logger.info("Starting processing request")
        
        try:
            # 1. SAM Inference
            image_np = np.array(image)
            logger.info("Generating mask with SAM...")
            mask_uint8 = self.sam.get_mask_from_points(image_np, points)
            mask_pil = Image.fromarray(mask_uint8).convert("L")
            
            # 2. ObjectClear Inference
            logger.info("Removing object with ObjectClear...")
            result_image = self.object_clear.remove_object(image, mask_pil)
            
            logger.info("Processing complete.")
            return result_image
            
        except CustomException as ce:
            raise ce
        except Exception as e:
            logger.error(f"Unexpected error in pipeline: {e}")
            raise CustomException("Pipeline processing failed", str(e))
