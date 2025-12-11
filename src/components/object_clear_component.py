import torch
from PIL import Image
from src.components.object_clear_pipeline import ObjectClearPipeline
from src.utils.image_utils import resize_by_short_side
from src.constants import OBJECT_CLEAR_REPO_ID, INFERENCE_STEPS, GUIDANCE_SCALE, DEFAULT_IMAGE_SIZE, USE_GPU
from src.logger import get_logger
from src.exceptions import CustomException

logger = get_logger(__name__)

class ObjectClearComponent:
    def __init__(self, device='cuda'):
        self.device = device
        
        logger.info(f"Loading ObjectClear model on {device}...")
        try:
            self.pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
                OBJECT_CLEAR_REPO_ID, # It will check local weights first due to our patch or cache
                torch_dtype=torch.float16 if 'cuda' in str(device) else torch.float32,
                apply_attention_guided_fusion=True,
                variant="fp16" if 'cuda' in str(device) else None,
            )
            
            if USE_GPU:
                # Force all to GPU
                self.pipe.to(device)
            else:
                # GPU/CPU Parallel (Model Offloading)
                if 'cuda' in str(device):
                     self.pipe.enable_model_cpu_offload()
                else:
                     self.pipe.to(device)

            logger.info(f"ObjectClear model loaded successfully. Strategy: {'Full GPU' if USE_GPU else 'CPU Offload'}")
        except Exception as e:
            logger.error(f"Failed to load ObjectClear model: {str(e)}")
            raise CustomException("Failed to load ObjectClear model", str(e))

    @torch.no_grad()
    def remove_object(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Removes object from image using mask.
        
        Args:
            image: PIL Image RGB
            mask: PIL Image L
            
        Returns:
            PIL Image RGB (inpainted)
        """
        try:
            generator = torch.Generator(device=self.device).manual_seed(42)
            
            # Resize logic
            image_for_inf = resize_by_short_side(image.copy(), DEFAULT_IMAGE_SIZE, resample=Image.BICUBIC)
            mask_for_inf = resize_by_short_side(mask, DEFAULT_IMAGE_SIZE, resample=Image.NEAREST)
            
            w, h = image_for_inf.size
            
            logger.info(f"Running inference with size {w}x{h}")
            
            result = self.pipe(
                prompt="remove the instance of object",
                image=image_for_inf,
                mask_image=mask_for_inf,
                generator=generator,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=h,
                width=w,
                return_attn_map=False,
            )
            
            fused_img = result.images[0]
            
            # Resize back to original size
            fused_img = fused_img.resize(image.size)
            
            return fused_img
            
        except Exception as e:
            logger.error(f"Error in ObjectClear inference: {str(e)}")
            raise CustomException("Error during object removal", str(e))
