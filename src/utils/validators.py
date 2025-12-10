import json
from src.exceptions import CustomException

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
