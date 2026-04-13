# backend/app/models/document.py
from sqlalchemy import Column, String, ForeignKey, Text, Integer, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base, TimestampMixin, UUIDMixin
import enum


class DocumentStatus(str, enum.Enum):
    """
    Lifecycle of an uploaded document:
    pending → processing → ready (searchable) or failed
    """
    PENDING    = "pending"
    PROCESSING = "processing"
    READY      = "ready"
    FAILED     = "failed"


class Document(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "documents"

    filename    = Column(String(500), nullable=False)
    file_type   = Column(String(50),  nullable=False)  # pdf, docx, txt
    file_size   = Column(Integer,     nullable=False)  # bytes
    file_path   = Column(String(500), nullable=False)  # local path or S3 key

    status      = Column(String(50), default=DocumentStatus.PENDING, nullable=False)
    error_msg   = Column(Text, nullable=True)  # if status=failed, why

    # How many chunks this doc was split into for RAG
    chunk_count = Column(Integer, default=0)

    # Foreign keys
    chatbot_id = Column(UUID(as_uuid=True), ForeignKey("chatbots.id",  ondelete="CASCADE"), nullable=False)
    tenant_id  = Column(UUID(as_uuid=True), ForeignKey("tenants.id",   ondelete="CASCADE"), nullable=False, index=True)

    # Relationships
    chatbot = relationship("Chatbot", back_populates="documents")
    tenant  = relationship("Tenant",  back_populates="documents")

    def __repr__(self):
        return f"<Document {self.filename} ({self.status})>"