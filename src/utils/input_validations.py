import json
from fastapi import UploadFile
from src.exceptions import CustomException
from src.constants import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB

def validate_coordinates(coords_json: str) -> list[list[int]]:
    """
    Parses and validates coordinates JSON string.
    Expected format: "[[x1, y1], [x2, y2], ...]"
    """
    try:
        points = json.loads(coords_json)
    except json.JSONDecodeError:
        raise CustomException("Invalid JSON for coords_json", "JSONDecodeError")
        
    if not isinstance(points, list):
         raise CustomException("coords_json must be a list of points", "InvalidFormat")
    
    for p in points:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            raise CustomException(f"Invalid point format: {p}. Expected [x, y]", "InvalidPoint")
            
    return points

async def validate_image_file(file: UploadFile):
    """
    Validates the input image file.
    Checks:
    1. Existence/Size > 0
    2. Extension
    3. MIME type
    4. File Size limit
    """
    # ---------1. Existing validation (implicitly handled by UploadFile, but check actual content later)
    # We can check filename existence at least
    if not file.filename:
        raise CustomException("File must have a filename", "InvalidFile")

    # ---------2. Extension validation
    filename = file.filename.lower()
    if '.' not in filename:
        raise CustomException("File must have an extension", "InvalidExtension")
    
    ext = filename.rsplit('.', 1)[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise CustomException(f"Invalid file extension: {ext}. Allowed: {ALLOWED_EXTENSIONS}", "InvalidExtension")

    # ---------3. MIME validation
    # This relies on the client sending the correct content-type, or we can use python-magic for stricter check.
    # For now, trusting Header + Extension check as detailed mime check requires reading the file which we do in app.py anyway.
    # We can do basic check here.
    if not file.content_type.startswith("image/"):
        raise CustomException(f"Invalid content type: {file.content_type}. Must be an image.", "InvalidMime")

    # ---------4. Size validation
    # UploadFile is a SpooledTemporaryFile. We can determine size by seeking.
    # Or determining from headers (content-length) if available, but checking actual size is safer.
    
    # Check Content-Length header if available as a quick fail
    # Note: size might not be populated in file.size immediately
    file.file.seek(0, 2) # Seek to end
    size = file.file.tell()
    file.file.seek(0) # Reset to beginning
    
    if size == 0:
        raise CustomException("File is empty", "EmptyFile")
        
    if size > MAX_FILE_SIZE_BYTES:
         raise CustomException(f"Image size shouldn't be greater that {MAX_FILE_SIZE_MB}mb", "FileSizeError")

    return True
