from fastapi.responses import JSONResponse
from typing import Any, Optional

class ResponseBuilder:
    """
    Utility class to standardize API responses.
    """

    @staticmethod
    def success(
        data: Any = None, 
        code: str = "SUCCESS_200", 
        message: str = "Request processed successfully"
    ) -> JSONResponse:
        """
        Returns a standardized success JSON response.
        """
        content = {
            "status": "success",
            "code": code,
            "message": message,
            "data": data
        }
        return JSONResponse(status_code=200, content=content)

    @staticmethod
    def error(
        code: str, 
        message: str, 
        details: Optional[Any] = None, 
        status_code: int = 500
    ) -> JSONResponse:
        """
        Returns a standardized error JSON response.
        """
        content = {
            "status": "error",
            "code": code,
            "message": message,
            "details": details
        }
        return JSONResponse(status_code=status_code, content=content)
