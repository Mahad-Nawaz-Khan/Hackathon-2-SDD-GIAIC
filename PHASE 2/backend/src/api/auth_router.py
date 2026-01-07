from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session
from typing import Dict, Any
from ..middleware.auth import get_current_user
from ..database import get_session
from ..services.auth_service import auth_service
from pydantic import BaseModel

# Initialize rate limiter for this router
limiter = Limiter(key_func=get_remote_address)


router = APIRouter(prefix="/api/v1", tags=["auth"])


class UserResponse(BaseModel):
    id: int
    clerk_user_id: str
    email: str
    first_name: str
    last_name: str


@router.get("/auth/me", response_model=UserResponse)
@limiter.limit("50/minute")  # 50 requests per minute for authenticated users
def get_current_user_info(
    request: Request,  # Required for rate limiting
    current_user: Dict[str, Any] = Depends(get_current_user),
    db_session: Session = Depends(get_session)
):
    """
    Get information about the currently authenticated user
    """
    clerk_user_id = auth_service.get_current_user_id(current_user)
    
    # Get user by Clerk user ID to get the integer user_id
    user = auth_service.get_user_by_clerk_id(clerk_user_id, db_session)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    return UserResponse(
        id=user.id,
        clerk_user_id=user.clerk_user_id,
        email=user.email,
        first_name=user.first_name or "",
        last_name=user.last_name or ""
    )