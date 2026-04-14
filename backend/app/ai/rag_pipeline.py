# backend/app/ai/rag_pipeline.py
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from app.core.config import settings
from app.ai.embeddings import embed_text
from app.models.document import Document
from typing import AsyncGenerator
import json


# The LLM — runs locally via Ollama
llm = ChatOllama(
    model=settings.LLM_MODEL,          # llama3.2
    base_url=settings.OLLAMA_BASE_URL,
    temperature=0.1,   # low = more factual, less creative — good for business bots
    streaming=True,
)


async def retrieve_relevant_chunks(
    query: str,
    chatbot_id: str,
    tenant_id: str,
    db: AsyncSession,
    top_k: int = 5
) -> list[dict]:
    """
    Find the most relevant document chunks for a query using vector similarity.
    Uses pgvector's <-> operator (cosine distance).
    """
    query_embedding = await embed_text(query)
    embedding_str = json.dumps(query_embedding)

    # pgvector similarity search — finds top_k closest vectors
    sql = text("""
        SELECT 
            content,
            filename,
            chunk_index,
            1 - (embedding <-> :embedding::vector) AS similarity
        FROM document_chunks
        WHERE chatbot_id = :chatbot_id
          AND tenant_id  = :tenant_id
        ORDER BY embedding <-> :embedding::vector
        LIMIT :top_k
    """)

    result = await db.execute(sql, {
        "embedding":  embedding_str,
        "chatbot_id": chatbot_id,
        "tenant_id":  tenant_id,
        "top_k":      top_k,
    })
    rows = result.fetchall()
    return [
        {"content": r.content, "filename": r.filename, "similarity": r.similarity}
        for r in rows
    ]


def build_prompt(system_prompt: str, context_chunks: list[dict], question: str) -> list:
    """
    Assembles the full prompt sent to the LLM:
    system prompt + retrieved context + user question.
    """
    if context_chunks:
        context_text = "\n\n---\n\n".join([
            f"Source: {c['filename']}\n{c['content']}"
            for c in context_chunks
        ])
        context_section = f"\n\nRelevant information from documents:\n{context_text}"
    else:
        context_section = "\n\nNo relevant documents found."

    return [
        SystemMessage(content=system_prompt + context_section),
        HumanMessage(content=question),
    ]


async def stream_chat_response(
    question: str,
    chatbot_id: str,
    tenant_id: str,
    system_prompt: str,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:
    """
    Full RAG pipeline — retrieves context then streams the AI response
    token by token back to the client.
    """
    # Step 1: find relevant chunks
    chunks = await retrieve_relevant_chunks(question, chatbot_id, tenant_id, db)

    # Step 2: build the prompt
    messages = build_prompt(system_prompt, chunks, question)

    # Step 3: stream response tokens
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content