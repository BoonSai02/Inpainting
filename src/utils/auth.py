import jwt
import os
from fastapi import Request, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.constants import ALPHA_JWT_SECRET_KEY as JWT_SECRET
from src.logger import get_logger

logger = get_logger("auth")
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """
    Verifies that the Bearer token is a valid JWT signed with our secret.
    """
    token = credentials.credentials
    
    if not JWT_SECRET:
         logger.error("JWT Secret is not configured.")
         raise HTTPException(status_code=500, detail="Authentication configuration error")

    try:
        # Decode and verify the token
        # Supabase uses HS256 by default.
        payload = jwt.decode(
            token, 
            JWT_SECRET, 
            algorithms=["HS256"],
            options={"verify_aud": False} # Verify audience if needed, usually 'authenticated'
        )
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired.")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except Exception as e:
         logger.error(f"Token verification error: {e}")
         raise HTTPException(status_code=401, detail="Authentication failed")
