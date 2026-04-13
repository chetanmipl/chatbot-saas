# backend/app/models/tenant.py
from sqlalchemy import Column, String, Boolean, Text
from sqlalchemy.orm import relationship
from app.db.base import Base, TimestampMixin, UUIDMixin
import secrets


class Tenant(Base, UUIDMixin, TimestampMixin):
    """
    A Tenant = one business/company that signed up to use your SaaS.
    Everything in the system belongs to a tenant.
    """
    __tablename__ = "tenants"

    name        = Column(String(255), nullable=False)
    slug        = Column(String(100), unique=True, nullable=False, index=True)
    # slug is a URL-friendly version of name e.g. "Acme Corp" → "acme-corp"

    plan        = Column(String(50), default="free", nullable=False)
    # Plans: free | starter | growth | enterprise

    is_active   = Column(Boolean, default=True, nullable=False)

    api_key     = Column(
        String(64),
        unique=True,
        nullable=False,
        default=lambda: secrets.token_urlsafe(32)
        # auto-generates a secure random API key for each tenant
    )

    # Business details
    website     = Column(String(255), nullable=True)
    logo_url    = Column(String(500), nullable=True)

    # Relationships — SQLAlchemy loads these automatically
    users       = relationship("User",     back_populates="tenant", cascade="all, delete-orphan")
    chatbots    = relationship("Chatbot",  back_populates="tenant", cascade="all, delete-orphan")
    documents   = relationship("Document", back_populates="tenant", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Tenant {self.name} ({self.plan})>"