# backend/app/db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings

# AsyncEngine — non-blocking DB calls, essential for FastAPI's async model
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,       # logs SQL in dev, disable in prod
    pool_size=10,              # max 10 persistent connections
    max_overflow=20,           # up to 20 extra under load
)

# Factory for creating DB sessions
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,    # don't re-fetch objects after commit
)


async def get_db():
    """FastAPI dependency — injects a DB session into route handlers."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise