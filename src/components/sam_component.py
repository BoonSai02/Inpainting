import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import os
from src.constants import SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE, USE_GPU
from src.logger import get_logger
from src.exceptions import CustomException

logger = get_logger(__name__)

class SAMComponent:
    def __init__(self, device='cuda'):
        self.device = device
        self.model_type = SAM_MODEL_TYPE
        
        if not os.path.exists(SAM_CHECKPOINT_PATH):
             logger.error(f"SAM checkpoint not found at {SAM_CHECKPOINT_PATH}")
             raise CustomException(f"SAM checkpoint not found. Please run src/utils/download_weights.py first.")

        logger.info(f"Loading SAM model from {SAM_CHECKPOINT_PATH} to {device}...")
        try:
            self.model = sam_model_registry[self.model_type](checkpoint=SAM_CHECKPOINT_PATH)
            
            if USE_GPU:
                self.model.to(device=self.device)
            else:
                # If offloading, keep on CPU initially, move to GPU only during inference
                # If device is cpu, it just stays on cpu
                if 'cuda' not in str(device):
                     self.model.to(device=self.device)
                
            self.predictor = SamPredictor(self.model)
            logger.info(f"SAM model loaded successfully. Strategy: {'Full GPU' if USE_GPU else 'manual offload'}")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {str(e)}")
            raise CustomException("Failed to load SAM model", str(e))

    @torch.no_grad()
    def get_mask_from_points(self, image_np: np.ndarray, points: list[list[int]], labels: list[int] = None) -> np.ndarray:
        """
        Generates a binary mask from point prompts.
        
        Args:
            image_np: (H, W, 3) RGB numpy array (0-255).
            points: List of [x, y] coordinates.
            labels: List of labels (1 for positive, 0 for negative). Defaults to all positive (1).
            
        Returns:
            np.ndarray: (H, W) binary mask (0 or 255) where 255 is the object.
        """
        try:
            self.predictor.set_image(image_np)
            
            points_np = np.array(points)
            if labels is None:
                labels_np = np.array([1] * len(points))
            else:
                labels_np = np.array(labels)
                
            # Predict masks
            
            # Manual Offloading Logic
            if not USE_GPU and 'cuda' in str(self.device):
                self.model.to(self.device)
                self.predictor.model = self.model # Ensure predictor uses the moved model
            
            masks, scores, logits = self.predictor.predict(
                point_coords=points_np,
                point_labels=labels_np,
                multimask_output=True 
            )
            
            if not USE_GPU and 'cuda' in str(self.device):
                self.model.to('cpu')
                torch.cuda.empty_cache()
            
            # Select the mask with the highest score
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx] # (H, W) boolean
            
            # Convert to uint8 0-255
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            
            self.predictor.reset_image()
            
            return mask_uint8
        except Exception as e:
            logger.error(f"Error in SAM inference: {str(e)}")
            raise CustomException("Error during SAM mask generation", str(e))
