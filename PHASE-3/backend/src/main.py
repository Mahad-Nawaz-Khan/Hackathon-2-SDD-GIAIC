from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
from sqlmodel import SQLModel, create_engine
from contextlib import asynccontextmanager
import logging
import os

# Load environment variables
load_dotenv(override=True)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./todo_app.db")

# Normalize Postgres URLs to a driver we actually ship (psycopg v3)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Create database engine
sql_echo_env = os.getenv("SQL_ECHO")
if sql_echo_env is None:
    sql_echo = DATABASE_URL.startswith("sqlite")
else:
    sql_echo = sql_echo_env.strip().lower() in {"1", "true", "yes", "on"}

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, echo=sql_echo, pool_pre_ping=True, connect_args=connect_args)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables on startup
    SQLModel.metadata.create_all(bind=engine)
    yield
    # Clean up on shutdown if needed

app = FastAPI(
    title="TODO API",
    description="API for the TODO application with Clerk authentication",
    version="1.0.0",
    lifespan=lifespan
)

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://hackathon-2-phase-2-pi.vercel.app", "http://localhost:3000"],  # Update with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware to add cache headers
@app.middleware("http")
async def add_cache_headers(request, call_next):
    response = await call_next(request)

    # Add cache headers for API responses
    if request.url.path.startswith("/api/"):
        # Don't cache user-specific data
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    else:
        # For other routes, set appropriate cache headers
        response.headers.setdefault("Cache-Control", "public, max-age=3600")

    return response

# Include API routes
from .api.task_router import router as task_router
from .api.auth_router import router as auth_router
from .api.tag_router import router as tag_router

app.include_router(task_router)
app.include_router(auth_router)
app.include_router(tag_router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the TODO API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)