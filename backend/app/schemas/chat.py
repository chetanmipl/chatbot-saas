# backend/app/schemas/chat.py
from pydantic import BaseModel
from uuid import UUID
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    message:    str
    session_id: str          # groups messages into one conversation
    chatbot_id: UUID


class MessageResponse(BaseModel):
    id:         UUID
    role:       str
    content:    str
    session_id: str
    created_at: datetime

    model_config = {"from_attributes": True}