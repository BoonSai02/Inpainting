from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
import torch
from PIL import Image
import io
import json
import numpy as np
import os
import sys

# Add src to sys.path to ensure imports work if run from parent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.segmentation.sam_handler import SAMHandler
from src.core.pipelines import ObjectClearPipeline
from src.core.utils import resize_by_short_side

# Global models
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading models on {device}...")
    
    # 1. Load SAM
    try:
        models['sam'] = SAMHandler(device=str(device))
    except Exception as e:
        print(f"Failed to load SAM: {e}")
        print("Did you run src/utils/download_weights.py?")
        raise e

    # 2. Load ObjectClear
    try:
        pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear", # It will check local weights first due to our patch
            torch_dtype=torch.float16 if 'cuda' in str(device) else torch.float32,
            apply_attention_guided_fusion=True,
            variant="fp16" if 'cuda' in str(device) else None,
        )
        pipe.to(device)
        models['object_clear'] = pipe
    except Exception as e:
        print(f"Failed to load ObjectClear: {e}")
        raise e
        
    print("All models loaded successfully.")
    yield
    # Clean up
    models.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.post("/inpainting/remove-object")
async def remove_object(
    file: UploadFile = File(...),
    coords_json: str = Form(...) # Expecting JSON string like "[[x1,y1], [x2,y2]]"
):
    try:
        # 1. Parse Inputs
        points = json.loads(coords_json)
        if not isinstance(points, list):
             raise HTTPException(status_code=400, detail="coords_json must be a list of points [[x,y], ...]")
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)
        
        # 2. Generate Mask (SAM)
        sam_handler = models['sam']
        # SAM expects points, we assume positive labels for all clicks for now
        mask_uint8 = sam_handler.get_mask_from_points(image_np, points)
        
        mask_pil = Image.fromarray(mask_uint8).convert("L") # ObjectClear expects PIL L/RGB
        
        # 3. Object Removal (ObjectClear)
        pipe = models['object_clear']
        generator = torch.Generator(device=pipe.device).manual_seed(42)
        
        # Resize logic similar to inference script: short side 512
        image_for_inf = resize_by_short_side(image.copy(), 512, resample=Image.BICUBIC)
        mask_for_inf = resize_by_short_side(mask_pil, 512, resample=Image.NEAREST)
        
        w, h = image_for_inf.size
        
        result = pipe(
            prompt="remove the instance of object",
            image=image_for_inf,
            mask_image=mask_for_inf,
            generator=generator,
            num_inference_steps=20, # Default
            guidance_scale=2.5,     # Default
            height=h,
            width=w,
            return_attn_map=False,
        )
        
        fused_img = result.images[0]
        
        # Resize back to original size
        fused_img = fused_img.resize(image.size)
        
        # 4. Return Response
        img_byte_arr = io.BytesIO()
        fused_img.save(img_byte_arr, format='PNG')
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON for coords_json")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
