# backend/app/models/chatbot.py
from sqlalchemy import Column, String, Boolean, ForeignKey, Text, JSON, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from app.db.base import Base, TimestampMixin, UUIDMixin


class Chatbot(Base, UUIDMixin, TimestampMixin):
    """
    One tenant can create multiple chatbots —
    e.g. "Support Bot", "Sales Bot", "HR Bot"
    each trained on different documents.
    """
    __tablename__ = "chatbots"

    name          = Column(String(255), nullable=False)
    description   = Column(Text, nullable=True)
    is_active     = Column(Boolean, default=True, nullable=False)
    domain = Column(String(100), nullable=True, default="general")

    # The system prompt shapes the bot's personality and scope
    system_prompt = Column(Text, default="""You are a helpful assistant. 
Answer questions based only on the provided documents. 
If you don't know the answer, say so clearly.""")

    # Widget customization — stored as JSON
    # e.g. {"color": "#4f6ef7", "position": "bottom-right", "greeting": "Hi!"}
    widget_config = Column(JSON, default=dict)

    # Limits per plan
    max_documents = Column(Integer, default=10)

    # Foreign key
    tenant_id = Column(UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False, index=True)

    # Relationships
    tenant    = relationship("Tenant",   back_populates="chatbots")
    documents = relationship("Document", back_populates="chatbot", cascade="all, delete-orphan")
    messages  = relationship("Message",  back_populates="chatbot", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Chatbot {self.name}>"