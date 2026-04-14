# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.db.session import engine
import app.db.registry  # noqa: F401, F403 - import all models for SQLAlchemy
from app.db.base import Base

from app.api.v1.endpoints.auth      import router as auth_router
from app.api.v1.endpoints.chatbots  import router as chatbots_router
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.chat      import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown."""
    # Startup: create all tables (Alembic handles prod migrations)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print(f"✅ {settings.APP_NAME} started — {settings.ENVIRONMENT} mode")
    yield
    # Shutdown: cleanup if needed
    await engine.dispose()
    print("👋 Server shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Industry-grade AI Chatbot SaaS API",
    docs_url="/docs" if settings.DEBUG else None,   # hide Swagger in production
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────
# Allows the React frontend (port 5173) to call the API (port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes (we'll add these phase by phase) ───────────────────────
# from app.api.v1 import router as api_router

app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
app.include_router(chatbots_router,  prefix=settings.API_V1_PREFIX)
app.include_router(documents_router, prefix=settings.API_V1_PREFIX)
app.include_router(chat_router,      prefix=settings.API_V1_PREFIX)

@app.get("/")
async def root():
    """Root endpoint — can be used for a quick sanity check."""
    return {"message": f"Welcome to {settings.APP_NAME} API!"}


@app.get("/health")
async def health_check():
    """Simple health check — used by Docker and load balancers."""
    return {"status": "ok", "environment": settings.ENVIRONMENT}