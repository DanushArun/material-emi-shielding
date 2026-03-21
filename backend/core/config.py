"""
Application configuration using Pydantic Settings
Reads from environment variables with fallback defaults
"""
from pydantic import field_validator
from pydantic_settings import BaseSettings
from typing import List, Union


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "EMI Shield Designer API"
    APP_VERSION: str = "4.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Security
    SECRET_KEY: str = "change-this-secret-key-in-env"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str = "postgresql://emi_user:emi_dev_password@localhost:5432/emi_shield_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600  # 1 hour

    # CORS (can be comma-separated string or list)
    CORS_ORIGINS: Union[List[str], str] = ["http://localhost:3002", "http://localhost:8001"]

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # Gemini AI (conversational design assistant)
    GEMINI_API_KEY: str = ""

    # Physics Calculation Limits
    MAX_FREQUENCY_POINTS: int = 1000
    MAX_THICKNESS_POINTS: int = 100

    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS from string or list"""
        if isinstance(v, str):
            # Split comma-separated string
            return [origin.strip() for origin in v.split(',')]
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
