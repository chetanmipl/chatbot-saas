# backend/app/api/v1/endpoints/chatbots.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.db.session import get_db
from app.core.dependencies import get_current_user, get_current_tenant
from app.models.user import User
from app.models.tenant import Tenant
from app.schemas.chatbot import ChatbotCreate, ChatbotUpdate, ChatbotResponse
from app.services.chatbot_service import (
    create_chatbot, get_chatbots, get_chatbot, update_chatbot, delete_chatbot
)

router = APIRouter(prefix="/chatbots", tags=["Chatbots"])


@router.post("", response_model=ChatbotResponse, status_code=201)
async def create(
    data: ChatbotCreate,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    return await create_chatbot(data, tenant, db)


@router.get("", response_model=list[ChatbotResponse])
async def list_chatbots(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    return await get_chatbots(tenant, db)


@router.get("/{chatbot_id}", response_model=ChatbotResponse)
async def get_one(
    chatbot_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    return await get_chatbot(chatbot_id, tenant, db)


@router.patch("/{chatbot_id}", response_model=ChatbotResponse)
async def update(
    chatbot_id: UUID,
    data: ChatbotUpdate,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    return await update_chatbot(chatbot_id, data, tenant, db)


@router.delete("/{chatbot_id}", status_code=204)
async def delete(
    chatbot_id: UUID,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db),
):
    await delete_chatbot(chatbot_id, tenant, db)