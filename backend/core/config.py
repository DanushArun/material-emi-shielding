"""
Application configuration using Pydantic Settings
Reads from environment variables with fallback defaults
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "EMI Shield Designer API"
    APP_VERSION: str = "4.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Security
    SECRET_KEY: str = "dev-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DATABASE_URL: str = "postgresql://emi_user:emi_dev_password@localhost:5432/emi_shield_db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600  # 1 hour

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # ML Model Settings
    ML_MODEL_PATH: str = "./models"
    ML_CACHE_ENABLED: bool = True

    # Physics Calculation Limits
    MAX_FREQUENCY_POINTS: int = 1000
    MAX_THICKNESS_POINTS: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()
