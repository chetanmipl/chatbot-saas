# backend/app/services/chatbot_service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
from uuid import UUID

from app.models.chatbot import Chatbot
from app.models.tenant import Tenant
from app.schemas.chatbot import ChatbotCreate, ChatbotUpdate


async def create_chatbot(data: ChatbotCreate, tenant: Tenant, db: AsyncSession) -> Chatbot:
    chatbot = Chatbot(
        name=data.name,
        description=data.description,
        system_prompt=data.system_prompt or """You are a helpful assistant. 
Answer questions based only on the provided documents. 
If you don't know the answer say: 'I don't have information about that.'""",
        widget_config=data.widget_config or {},
        tenant_id=tenant.id,
    )
    db.add(chatbot)
    await db.commit()
    await db.refresh(chatbot)
    return chatbot


async def get_chatbots(tenant: Tenant, db: AsyncSession) -> list[Chatbot]:
    """Get all chatbots belonging to this tenant only."""
    result = await db.execute(
        select(Chatbot)
        .where(Chatbot.tenant_id == tenant.id)
        .order_by(Chatbot.created_at.desc())
    )
    return result.scalars().all()


async def get_chatbot(chatbot_id: UUID, tenant: Tenant, db: AsyncSession) -> Chatbot:
    """Get a single chatbot — enforces tenant isolation."""
    result = await db.execute(
        select(Chatbot).where(
            Chatbot.id == chatbot_id,
            Chatbot.tenant_id == tenant.id  # ← can never access another tenant's bot
        )
    )
    chatbot = result.scalar_one_or_none()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")
    return chatbot


async def update_chatbot(chatbot_id: UUID, data: ChatbotUpdate, tenant: Tenant, db: AsyncSession) -> Chatbot:
    chatbot = await get_chatbot(chatbot_id, tenant, db)
    update_data = data.model_dump(exclude_unset=True)  # only update provided fields
    for field, value in update_data.items():
        setattr(chatbot, field, value)
    await db.commit()
    await db.refresh(chatbot)
    return chatbot


async def delete_chatbot(chatbot_id: UUID, tenant: Tenant, db: AsyncSession):
    chatbot = await get_chatbot(chatbot_id, tenant, db)
    await db.delete(chatbot)
    await db.commit()