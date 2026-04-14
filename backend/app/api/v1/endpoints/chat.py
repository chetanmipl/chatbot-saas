# backend/app/api/v1/endpoints/chat.py
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.core.dependencies import get_current_tenant
from app.models.tenant import Tenant
from app.models.chatbot import Chatbot
from app.models.message import Message
from app.schemas.chat import ChatRequest
from app.ai.rag_pipeline import stream_chat_response
from fastapi import HTTPException

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/stream")
async def chat_stream(
    data:   ChatRequest,
    tenant: Tenant       = Depends(get_current_tenant),
    db:     AsyncSession = Depends(get_db),
):
    """
    Streams the AI response token by token using Server-Sent Events.
    The frontend receives text progressively — like ChatGPT's typing effect.
    """
    # Verify chatbot belongs to this tenant
    result = await db.execute(
        select(Chatbot).where(
            Chatbot.id        == data.chatbot_id,
            Chatbot.tenant_id == tenant.id,
            Chatbot.is_active == True,
        )
    )
    chatbot = result.scalar_one_or_none()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    # Save user message
    user_msg = Message(
        role="user",
        content=data.message,
        session_id=data.session_id,
        chatbot_id=chatbot.id,
        tenant_id=tenant.id,
    )
    db.add(user_msg)
    await db.commit()

    # Collect full response to save to DB after streaming
    full_response = []

    async def generate():
        async for token in stream_chat_response(
            question=data.message,
            chatbot_id=str(chatbot.id),
            tenant_id=str(tenant.id),
            system_prompt=chatbot.system_prompt,
            db=db,
        ):
            full_response.append(token)
            yield token

        # Save bot response after streaming completes
        bot_msg = Message(
            role="assistant",
            content="".join(full_response),
            session_id=data.session_id,
            chatbot_id=chatbot.id,
            tenant_id=tenant.id,
        )
        db.add(bot_msg)
        await db.commit()

    return StreamingResponse(generate(), media_type="text/plain")


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    tenant:     Tenant       = Depends(get_current_tenant),
    db:         AsyncSession = Depends(get_db),
):
    """Get all messages for a conversation session."""
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id, Message.tenant_id == tenant.id)
        .order_by(Message.created_at.asc())
    )
    messages = result.scalars().all()
    return [
        {"role": m.role, "content": m.content, "created_at": m.created_at}
        for m in messages
    ]