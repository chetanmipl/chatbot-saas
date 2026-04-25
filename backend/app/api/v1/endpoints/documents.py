# backend/app/api/v1/endpoints/documents.py
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID

from app.db.session import get_db
from app.core.dependencies import get_current_tenant
from app.models.tenant import Tenant
from app.models.document import Document
from app.services.document_service_ import upload_document

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", status_code=201)
async def upload(
    file:       UploadFile = File(...),
    chatbot_id: str        = Form(...),
    tenant:     Tenant     = Depends(get_current_tenant),
    db:         AsyncSession = Depends(get_db),
):
    doc = await upload_document(file, chatbot_id, tenant, db)
    return {
        "id":          str(doc.id),
        "filename":    doc.filename,
        "status":      doc.status,
        "chunk_count": doc.chunk_count,
        "file_size":   doc.file_size,
    }


@router.get("/chatbot/{chatbot_id}")
async def list_documents(
    chatbot_id: UUID,
    tenant:     Tenant       = Depends(get_current_tenant),
    db:         AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Document).where(
            Document.chatbot_id == chatbot_id,
            Document.tenant_id  == tenant.id
        ).order_by(Document.created_at.desc())
    )
    docs = result.scalars().all()
    return [
        {
            "id":          str(d.id),
            "filename":    d.filename,
            "status":      d.status,
            "chunk_count": d.chunk_count,
            "file_type":   d.file_type,
            "file_size":   d.file_size,
            "created_at":  d.created_at,
        }
        for d in docs
    ]