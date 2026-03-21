"""
EMI Shield Designer - FastAPI Backend
Main application entry point
"""
import sys
import os

# Add project root to path so `src.*` imports work when running from backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
from typing import Dict, Any

from core.config import settings
from api.v1.routes import physics, auth, materials, analysis, chat, multilayer, composites, advanced

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered electromagnetic interference shielding design platform",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add X-Process-Time header to all responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Custom validation error handler"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "message": "Invalid request data"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "message": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


# Root endpoint
@app.get("/", tags=["System"])
async def root() -> Dict[str, str]:
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


# API v1 routes
app.include_router(physics.router, prefix="/api/v1/physics", tags=["Physics"])
app.include_router(materials.router, prefix="/api/v1/materials", tags=["Materials"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["Analysis"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["AI Chat"])
app.include_router(multilayer.router, prefix="/api/v1/multilayer", tags=["Multilayer"])
app.include_router(composites.router, prefix="/api/v1/composites", tags=["Composites"])
app.include_router(advanced.router, prefix="/api/v1/advanced", tags=["Advanced"])

# Auth router - optional, requires database
try:
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
except Exception as e:
    import logging
    logging.getLogger(__name__).warning(f"Auth router not loaded: {e}")


# Lifecycle (modern FastAPI pattern, replaces deprecated @app.on_event)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"API docs: http://localhost:8001/docs")
    yield
    print(f"Shutting down {settings.APP_NAME}")

app.router.lifespan_context = lifespan


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.DEBUG
    )
