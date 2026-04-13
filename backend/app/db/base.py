# backend/app/db/base.py
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, DateTime, func
import uuid
from sqlalchemy.dialects.postgresql import UUID


class Base(DeclarativeBase):
    """All models inherit from this — gives every table id + timestamps."""
    pass


class TimestampMixin:
    """Re-usable mixin that adds created_at / updated_at to any model."""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class UUIDMixin:
    """Re-usable mixin that adds id (UUID) to any model."""
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,server_default=func.gen_random_uuid())
    