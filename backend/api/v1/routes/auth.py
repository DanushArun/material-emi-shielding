"""
Authentication endpoints
Handles user registration, login, and token management
"""
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr, Field, validator
from typing import Dict, Any
import re

from core.security import (
    create_access_token,
    create_refresh_token,
    verify_password,
    get_password_hash,
    decode_token
)

router = APIRouter()


# Request/Response Models
class UserRegistration(BaseModel):
    """User registration request"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=255)

    @validator('username')
    def validate_username(cls, v):
        """Ensure username contains only alphanumeric and underscores"""
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError('Username must contain only letters, numbers, and underscores')
        return v

    @validator('password')
    def validate_password(cls, v):
        """Ensure password meets security requirements"""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one number')
        return v

    class Config:
        schema_extra = {
            "example": {
                "email": "engineer@company.com",
                "username": "john_doe",
                "password": "SecurePass123",
                "full_name": "John Doe"
            }
        }


class UserLogin(BaseModel):
    """User login request"""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


# Endpoints
@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegistration) -> Dict[str, Any]:
    """
    Register a new user account

    Creates a new user with the free subscription tier.
    Returns access and refresh tokens for immediate login.
    """
    # TODO: Implement database integration
    # For now, return mock response for testing

    # In production, this would:
    # 1. Check if email/username already exists
    # 2. Hash the password
    # 3. Insert user into database
    # 4. Generate tokens
    # 5. Return user data and tokens

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Database integration pending. User registration will be available once PostgreSQL connection is established."
    )


@router.post("/login", response_model=TokenResponse)
async def login_user(credentials: UserLogin) -> Dict[str, Any]:
    """
    Authenticate user and return tokens

    Validates credentials against database and returns
    access token (30 min) and refresh token (7 days).
    """
    # TODO: Implement database integration
    # For now, return mock response for testing

    # Mock admin user for testing (matches init.sql)
    if credentials.email == "admin@emishield.com" and credentials.password == "admin123":
        # Create tokens
        user_data = {
            "sub": "00000000-0000-0000-0000-000000000000",  # Mock UUID
            "email": credentials.email,
            "subscription_tier": "enterprise"
        }

        access_token = create_access_token(user_data)
        refresh_token = create_refresh_token(user_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": user_data["sub"],
                "email": credentials.email,
                "username": "admin",
                "full_name": "System Administrator",
                "subscription_tier": "enterprise",
                "is_verified": True
            }
        }

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password"
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(request: RefreshTokenRequest) -> Dict[str, Any]:
    """
    Refresh access token using refresh token

    Generates a new access token from a valid refresh token.
    Useful for maintaining sessions without re-login.
    """
    try:
        # Decode refresh token
        payload = decode_token(request.refresh_token)

        # Extract user data
        user_data = {
            "sub": payload.get("sub"),
            "email": payload.get("email"),
            "subscription_tier": payload.get("subscription_tier", "free")
        }

        # Generate new tokens
        new_access_token = create_access_token(user_data)
        new_refresh_token = create_refresh_token(user_data)

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "user": {
                "id": user_data["sub"],
                "email": user_data["email"],
                "subscription_tier": user_data["subscription_tier"]
            }
        }

    except HTTPException as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )


@router.post("/logout")
async def logout_user() -> Dict[str, str]:
    """
    Logout user (client-side token deletion)

    In a stateless JWT system, logout is handled client-side
    by deleting the stored tokens. This endpoint is provided
    for API consistency.

    For production, implement token blacklisting with Redis.
    """
    return {
        "message": "Logout successful. Please delete tokens from client storage."
    }


@router.get("/me")
async def get_current_user() -> Dict[str, Any]:
    """
    Get current user profile

    Returns profile information for the authenticated user.
    Requires valid access token in Authorization header.
    """
    # TODO: Implement with database
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Database integration pending"
    )
