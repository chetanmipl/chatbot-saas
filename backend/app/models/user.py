# backend/app/models/user.py
from sqlalchemy import Column, String, Boolean, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base, TimestampMixin, UUIDMixin
import enum


class UserRole(str, enum.Enum):
    """
    Roles inside a tenant:
    - owner:  the person who signed up — full access
    - admin:  can manage bots and team members
    - member: can only view and chat
    """
    OWNER  = "owner"
    ADMIN  = "admin"
    MEMBER = "member"


class User(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    email           = Column(String(255), unique=True, nullable=False, index=True)
    full_name       = Column(String(255), nullable=False)
    hashed_password = Column(String(255), nullable=False)

    role            = Column(String(50), default=UserRole.OWNER, nullable=False)
    is_active       = Column(Boolean, default=True, nullable=False)
    is_verified     = Column(Boolean, default=False, nullable=False)
    # is_verified: email verification — we'll add email later

    # Foreign key — which company this user belongs to
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)

    # Relationships
    tenant   = relationship("Tenant", back_populates="users")

    def __repr__(self):
        return f"<User {self.email} ({self.role})>"