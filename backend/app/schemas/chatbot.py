# backend/app/schemas/chatbot.py
from pydantic import BaseModel
from uuid import UUID
from typing import Optional
from datetime import datetime


class ChatbotCreate(BaseModel):
    name:          str
    description:   Optional[str] = None
    system_prompt: Optional[str] = None
    widget_config: Optional[dict] = {}


class ChatbotUpdate(BaseModel):
    name:          Optional[str] = None
    description:   Optional[str] = None
    system_prompt: Optional[str] = None
    widget_config: Optional[dict] = None
    is_active:     Optional[bool] = None


class ChatbotResponse(BaseModel):
    id:            UUID
    name:          str
    description:   Optional[str]
    system_prompt: str
    widget_config: dict
    is_active:     bool
    tenant_id:     UUID
    created_at:    datetime

    model_config = {"from_attributes": True}