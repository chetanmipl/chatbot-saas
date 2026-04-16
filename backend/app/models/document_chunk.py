# backend/app/models/document_chunk.py
from sqlalchemy import Column, String, Integer, Text, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.db.base import Base, TimestampMixin, UUIDMixin


class DocumentChunk(Base, UUIDMixin, TimestampMixin):
    """
    One document gets split into many chunks.
    Each chunk has its text + its vector embedding stored side by side.
    """
    __tablename__ = "document_chunks"

    content     = Column(Text, nullable=False)       # the actual text
    chunk_index = Column(Integer, nullable=False)    # position in original doc
    filename    = Column(String(500), nullable=False) # for showing source in UI

    # The vector embedding — 768 dimensions for nomic-embed-text
    embedding   = Column(Vector(768), nullable=True)
    # The vector embedding — 768 dimensions for hugging-face sentence-transformers/all-MiniLM-L6-v2
    #embedding = Column(Vector(384), nullable=True)

    # Foreign keys for isolation
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chatbot_id  = Column(UUID(as_uuid=True), ForeignKey("chatbots.id",  ondelete="CASCADE"), nullable=False)
    tenant_id   = Column(UUID(as_uuid=True), ForeignKey("tenants.id",   ondelete="CASCADE"), nullable=False)

    # Index for fast vector search — ivfflat is pgvector's approximate search index
    __table_args__ = (
        Index(
            "ix_chunks_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )