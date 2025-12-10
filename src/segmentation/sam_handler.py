import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
import os

class SAMHandler:
    def __init__(self, device='cuda', model_type='vit_h'):
        self.device = device
        self.model_type = model_type
        
        # Path to weights
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # src/segmentation -> src/weights
        weights_dir = os.path.join(current_dir, '..', 'weights')
        # We will enforce this filename in the download script
        checkpoint_path = os.path.join(weights_dir, "sam_vit_h_4b8939.pth")
        
        if not os.path.exists(checkpoint_path):
             raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}. Please run src/utils/download_weights.py first.")

        print(f"Loading SAM model from {checkpoint_path} to {device}...")
        self.model = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        print("SAM model loaded.")

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
        self.predictor.set_image(image_np)
        
        points_np = np.array(points)
        if labels is None:
            labels_np = np.array([1] * len(points))
        else:
            labels_np = np.array(labels)
            
        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=True 
        )
        
        # Select the mask with the highest score
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx] # (H, W) boolean
        
        # Convert to uint8 0-255
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        
        self.predictor.reset_image()
        
        return mask_uint8
