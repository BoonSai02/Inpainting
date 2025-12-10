from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from PIL import Image
import io
import json
import uvicorn
import os
import sys

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.pipeline import InpaintingPipeline
from src.exceptions import CustomException
from src.logger import get_logger
from src.utils.input_validations import validate_coordinates, validate_image_file

logger = get_logger("app")

# Global pipeline instance
pipeline_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance
    try:
        pipeline_instance = InpaintingPipeline()
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pass 
    yield
    # Clean up if needed
    if pipeline_instance:
        del pipeline_instance

app = FastAPI(lifespan=lifespan)

@app.post("/inpainting/remove-object")
async def remove_object(
    file: UploadFile = File(...),
    coords_json: str = Form(...) 
):
    global pipeline_instance
    if pipeline_instance is None:
         raise HTTPException(status_code=500, detail="Model pipeline not initialized. Check server logs.")

    try:
        # 1. Parse Inputs
        points = validate_coordinates(coords_json)
        
        # 2. Validate Image
        await validate_image_file(file)
        
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # 2. Process via Pipeline
        result_image = pipeline_instance.process_request(image, points)
        
        # 3. Return Response
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except CustomException as ce:
        logger.error(f"Custom Exception: {ce}")
        raise HTTPException(status_code=500, detail=str(ce))
    except Exception as e:
        logger.error(f"Unhandled Exception: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
