# backend/app/services/document_service.py
import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.document import Document, DocumentStatus
from app.models.document_chunk import DocumentChunk
from app.models.chatbot import Chatbot
from app.models.tenant import Tenant
from app.ai.embeddings import embed_texts
from app.core.config import settings


ALLOWED_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 80) -> list[str]:
    """
    Improved chunking:
    - Smaller chunks (300 vs 500) = more precise retrieval
    - Higher overlap (80 vs 50) = less info lost at boundaries
    - Sentence-aware splitting = no mid-sentence cuts
    """
    import re

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Try to split on sentence boundaries first
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            # Start new chunk WITH overlap from end of previous chunk
            if current:
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = overlap_text + " " + sentence
            else:
                current = sentence

    if current.strip():
        chunks.append(current.strip())

    # Filter out tiny chunks that carry no meaning
    return [c for c in chunks if len(c.split()) > 8]  # skip tiny chunks


def extract_text(file_path: Path, file_type: str) -> str:
    """Extract plain text from PDF, DOCX or TXT files."""
    if file_type == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif file_type == "docx":
        from docx import Document as DocxDocument

        doc = DocxDocument(str(file_path))
        return "\n".join(p.text for p in doc.paragraphs)

    elif file_type == "txt":
        return file_path.read_text(encoding="utf-8")

    raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")


async def upload_document(
    file: UploadFile,
    chatbot_id: str,
    tenant: Tenant,
    db: AsyncSession,
) -> Document:
    """Save file, parse it, chunk it, embed it, store in pgvector."""

    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400, detail="Only PDF, DOCX, and TXT files are supported"
        )

    file_type = ALLOWED_TYPES[file.content_type]

    # Validate chatbot belongs to this tenant
    result = await db.execute(
        select(Chatbot).where(Chatbot.id == chatbot_id, Chatbot.tenant_id == tenant.id)
    )
    chatbot = result.scalar_one_or_none()
    if not chatbot:
        raise HTTPException(status_code=404, detail="Chatbot not found")

    # Save file to disk
    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}_{file.filename}"

    async with aiofiles.open(save_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Create document record
    document = Document(
        filename=file.filename,
        file_type=file_type,
        file_size=len(content),
        file_path=str(save_path),
        status=DocumentStatus.PROCESSING,
        chatbot_id=chatbot_id,
        tenant_id=tenant.id,
    )
    db.add(document)
    await db.flush()  # get document.id

    try:
        # Extract text
        text = extract_text(save_path, file_type)
        if not text.strip():
            raise ValueError("No text could be extracted from the file")

        # Split into chunks
        chunks = chunk_text(text)

        # Embed all chunks in one batch call (much faster than one-by-one)
        embeddings = await embed_texts(chunks)

        # Store each chunk + its embedding
        for i, (chunk_text_content, embedding) in enumerate(zip(chunks, embeddings)):
            chunk = DocumentChunk(
                content=chunk_text_content,
                chunk_index=i,
                filename=file.filename,
                embedding=embedding,
                document_id=document.id,
                chatbot_id=chatbot_id,
                tenant_id=tenant.id,
            )
            db.add(chunk)

        document.status = DocumentStatus.READY
        document.chunk_count = len(chunks)

    except Exception as e:
        document.status = DocumentStatus.FAILED
        document.error_msg = str(e)

    await db.commit()
    await db.refresh(document)
    return document
