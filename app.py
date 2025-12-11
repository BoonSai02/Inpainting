# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import io
import os
import sys
import time
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

# Add src to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.pipeline import InpaintingPipeline
from src.exceptions import CustomException
from src.logger import get_logger
from src.utils.input_validations import validate_coordinates, validate_image_file
from src.utils.response_builder import ResponseBuilder
from src.constants import ResponseCode, MODEL_TTL_SECONDS

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
logger = get_logger("app")

# -----------------------------------------------------------------------------
# Background Tasks
# -----------------------------------------------------------------------------
async def resource_monitor(app: FastAPI):
    """
    Background task to monitor API activity and release resources if inactive.
    """
    logger.info(f"Resource monitor started. TTL: {MODEL_TTL_SECONDS}s")
    while True:
        try:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            # Calculate inactivity duration
            elapsed = time.time() - app.state.last_activity
            
            if elapsed > MODEL_TTL_SECONDS:
                # If pipeline has loaded models (we check a flag or just call cleanup)
                # Calling cleanup is safe as it checks internal state
                if hasattr(app.state, 'pipeline'):
                    # We can opt to only log if we actually cleaned something up,
                    # but the pipeline.cleanup() logs internally.
                    # To avoid spamming logs, we might check if cleanup is needed?
                    # For now, we rely on the pipeline.cleanup logic being idempotent and fast if empty.
                    
                    # Optimization: Only call cleanup if we think it's loaded to avoid log spam
                    # But checking internal state breaks encapsulation. 
                    # Let's assume pipeline.cleanup() is cheap.
                    
                    # Actually, if we want to avoid spamming "Cleaning up..." every 10s after it's already clean:
                    # We can check if *sam* or *object_clear* is not None before calling.
                    if app.state.pipeline.sam is not None or app.state.pipeline.object_clear is not None:
                        logger.info(f"Inactivity limit reached ({elapsed:.0f}s > {MODEL_TTL_SECONDS}s). Releasing resources.")
                        app.state.pipeline.cleanup()
                        
        except asyncio.CancelledError:
            logger.info("Resource monitor task cancelled.")
            break
        except Exception as e:
            logger.error(f"Error in resource monitor: {e}")

# -----------------------------------------------------------------------------
# Lifespan Management
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of the application.
    Initialize the pipeline on startup and clean up resources on shutdown.
    """
    # Startup
    logger.info("Starting up application...")
    
    # Initialize State
    app.state.last_activity = time.time()
    
    try:
        app.state.pipeline = InpaintingPipeline()
        logger.info("Pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pass

    # Start Background Monitor
    monitor_task = asyncio.create_task(resource_monitor(app))

    yield

    # Shutdown
    logger.info("Shutting down application...")
    
    # Cancel monitor
    monitor_task.cancel()
    try:
        await monitor_task
    except asyncio.CancelledError:
        pass
        
    if hasattr(app.state, 'pipeline'):
        app.state.pipeline.cleanup()
        logger.info("Pipeline resources cleaned up.")

# -----------------------------------------------------------------------------
# App Initialization
# -----------------------------------------------------------------------------
app = FastAPI(
    title="ObjectClear API",
    description="API for removing objects from images using SAM and ObjectClear",
    version="1.0.0",
    lifespan=lifespan
)

# -----------------------------------------------------------------------------
# Middleware
# -----------------------------------------------------------------------------
# Add CORS Middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now. Secure this in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Exception Handlers
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {exc}")
    return ResponseBuilder.error(
        code=ResponseCode.ERR_PROCESSING_FAILED,
        message="Internal Server Error",
        details=str(exc)
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    return ResponseBuilder.error(
        code=ResponseCode.ERR_PROCESSING_FAILED,
        message=exc.detail,
        status_code=exc.status_code
    )

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    if not hasattr(app.state, 'pipeline'):
        return ResponseBuilder.error(
            code=ResponseCode.ERR_MODEL_NOT_READY,
            message="Service unhealthy: Pipeline not initialized",
            status_code=503
        )
    return ResponseBuilder.success(message="Service is healthy")


@app.post("/inpainting/remove-object")
async def remove_object(
    request: Request,
    file: UploadFile = File(...),
    coords_json: str = Form(...)
):
    """
    Endpoint to remove objects from an image based on provided coordinates.
    """
    # 0. Update Activity (Heartbeat)
    request.app.state.last_activity = time.time()
    
    # Check if pipeline is ready
    if not hasattr(request.app.state, 'pipeline'):
        logger.error("Pipeline not initialized in app state.")
        return ResponseBuilder.error(
            code=ResponseCode.ERR_MODEL_NOT_READY,
            message="Model pipeline not initialized. Check server logs.",
            status_code=503
        )
    
    pipeline = request.app.state.pipeline

    try:
        # 1. Parse Inputs (CPU bound, fast)
        try:
            points = validate_coordinates(coords_json)
        except ValueError as ve:
             return ResponseBuilder.error(
                code=ResponseCode.ERR_INVALID_INPUT,
                message="Invalid coordinates format",
                details=str(ve),
                status_code=400
            )

        # 2. Validate Image (I/O bound)
        try:
            await validate_image_file(file)
        except ValueError as ve:
             return ResponseBuilder.error(
                code=ResponseCode.ERR_INVALID_INPUT,
                message="Invalid image file",
                details=str(ve),
                status_code=400
            )

        # Read image
        try:
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except UnidentifiedImageError:
            logger.error(f"UnidentifiedImageError: File pretending to be an image could not be opened.")
            return ResponseBuilder.error(
                code=ResponseCode.ERR_INVALID_INPUT,
                message="Invalid image content",
                status_code=400
            )

        # 3. Process via Pipeline (Blocking / GPU bound - Run in threadpool)
        # We use run_in_threadpool to ensure the main event loop is not blocked
        try:
            result_image = await run_in_threadpool(pipeline.process_request, image, points)
        except CustomException as ce:
            logger.error(f"Custom Exception in pipeline: {ce}")
            return ResponseBuilder.error(
                code=ResponseCode.ERR_PROCESSING_FAILED,
                message="Processing failed",
                details=str(ce),
                status_code=500
            )
        
        # 4. Return Response
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        logger.error(f"Unhandled Exception in endpoint: {e}")
        return ResponseBuilder.error(
            code=ResponseCode.ERR_PROCESSING_FAILED,
            message="Internal Server Error",
            details=str(e)
        )

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
