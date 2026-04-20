# backend/app/ai/rag_pipeline.py
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from app.core.config import settings
import numpy as np
from app.ai.embeddings import embed_text, embed_texts
from app.models.document import Document
from typing import AsyncGenerator
import json
from langchain_groq import ChatGroq


# The LLM — runs locally via Ollama
# llm = ChatOllama(
#     model=settings.LLM_MODEL,          # llama3.2
#     base_url=settings.OLLAMA_BASE_URL,
#     temperature=0.1,   # low = more factual, less creative — good for business bots
#     streaming=True,
# )

llm = ChatGroq(
    model=settings.GROQ_MODEL,
    api_key=settings.GROQ_API_KEY,
    temperature=0.0,  # ← was 0.1, now 0 = fully deterministic, no creativity
    max_tokens=1024,  # max tokens in response
    # repeat_penalty=1.1,    # reduces repetition
)


async def generate_hypothetical_answer(question: str) -> str:
    """
    Ask LLM to write a fake answer as if it exists in a document.
    We embed THIS instead of the raw question.
    """
    hyde_prompt = [
        SystemMessage(content="""Write a short factual passage (2-3 sentences) 
that would directly answer the following question, as if you are writing 
a section from an official document or article. 
Do not say 'I think' or 'probably'. Just write the passage directly."""),
        HumanMessage(content=question)
    ]
    response = await llm.ainvoke(hyde_prompt)
    return response.content


async def retrieve_relevant_chunks(
    query: str,
    chatbot_id: str,
    tenant_id: str,
    db: AsyncSession,
    top_k: int = 8
) -> list[dict]:

    # ── Step 1: HyDE — embed hypothetical answer, not raw query ──
    # hypothetical = await generate_hypothetical_answer(query)
    # print(f"\n💭 HyDE hypothetical: {hypothetical[:150]}...")

    # # Embed both — use average for robustness
    # query_emb       = await embed_text(query)
    # hypothetical_emb = await embed_text(hypothetical)

    # # Average the two embeddings
    # import numpy as np
    # combined_emb  = np.mean([query_emb, hypothetical_emb], axis=0).tolist()
    # embedding_str = "[" + ",".join(str(x) for x in combined_emb) + "]"

    # ── Step 1A: Query Variations Embedding ─────────────────────
    queries = await generate_query_variations(query)
    print(f"  Query variations: {queries}")

    query_embeddings = await embed_texts(queries)
    query_combined_emb = np.mean(query_embeddings, axis=0).tolist()
    query_embedding_str = "[" + ",".join(str(x) for x in query_combined_emb) + "]"


    # ── Step 1B: HyDE Embedding (separate, NOT averaged) ────────
    use_hyde = len(query.split()) > 4  # only for complex queries

    if use_hyde:
        hypothetical = await generate_hypothetical_answer(query)
        print(f"\n💭 HyDE hypothetical: {hypothetical[:150]}...")
        hyde_embedding = await embed_text(hypothetical)
        hyde_embedding_str = "[" + ",".join(str(x) for x in hyde_embedding) + "]"
    else:
        hyde_embedding_str = None

    # print(f"\n💭 HyDE hypothetical: {hypothetical[:150]}...")

    # hyde_embedding = await embed_text(hypothetical)
    # hyde_embedding_str = "[" + ",".join(str(x) for x in hyde_embedding) + "]"

    # queries = await generate_query_variations(query)
    # embeddings = await embed_texts(queries)
    # combined_emb = np.mean(embeddings, axis=0)
    # embedding_str = "[" + ",".join(str(x) for x in combined_emb) + "]"

    

    # ── Step 2: Vector search ──────────────────────────────────────
    vector_sql = text("""
        SELECT
            content,
            filename,
            chunk_index,
            embedding::text,
            1 - (embedding <-> CAST(:embedding AS vector)) AS similarity,
            'vector' AS source
        FROM document_chunks
        WHERE chatbot_id = CAST(:chatbot_id AS uuid)
          AND tenant_id  = CAST(:tenant_id  AS uuid)
        ORDER BY embedding <-> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    # ── Step 3: Keyword search ────────────────────────────────────
    keyword_sql = text("""
        SELECT
            content,
            filename,
            chunk_index,
            embedding::text,
            ts_rank(content_tsv, plainto_tsquery('english', :query)) AS similarity,
            'keyword' AS source
        FROM document_chunks
        WHERE chatbot_id  = CAST(:chatbot_id AS uuid)
          AND tenant_id   = CAST(:tenant_id  AS uuid)
          AND content_tsv @@ plainto_tsquery('english', :query)
        ORDER BY similarity DESC
        LIMIT :top_k
    """)

    params = {
        "embedding":  query_embedding_str,
        "chatbot_id": chatbot_id,
        "tenant_id":  tenant_id,
        "top_k":      top_k,
        "query":      query,
    }

    # Query variation search
    vector_rows_query = (await db.execute(
        vector_sql,
        {**params, "embedding": query_embedding_str}
    )).fetchall()

    # HyDE search
    vector_rows_hyde = []
    if hyde_embedding_str:
        vector_rows_hyde = (await db.execute(
            vector_sql,
            {**params, "embedding": hyde_embedding_str}
        )).fetchall()

    print(f"  Vector (query) hits: {len(vector_rows_query)}")
    print(f"  Vector (HyDE) hits: {len(vector_rows_hyde)}")
    
    keyword_rows = (await db.execute(keyword_sql, params)).fetchall()

    print(f"  Vector hits: {len(vector_rows_query) + len(vector_rows_hyde)} | Keyword hits: {len(keyword_rows)}")

    # ── Step 4: Merge + deduplicate ───────────────────────────────
    seen   = {}

    def add_result(r, source_label: str):
        key     = (r.filename, r.chunk_index)
        raw_sim = float(r.similarity)

        try:
            emb = json.loads(r.embedding) if r.embedding else []
        except Exception:
            emb = []

        if key in seen:
            # Keep highest similarity, just tag the source
            if raw_sim > seen[key]["similarity"]:
                seen[key]["similarity"] = raw_sim
            seen[key]["source"] += f"+{source_label}"
        else:
            seen[key] = {
                "content":    r.content,
                "filename":   r.filename,
                "similarity": raw_sim,   # ← raw score, no weighting
                "embedding":  emb,
                "source":     source_label,
            }

    for r in vector_rows_query:
        add_result(r, "vector")

    for r in vector_rows_hyde:
        add_result(r, "hyde")

    for r in keyword_rows:
        add_result(r, "keyword")

    merged = sorted(seen.values(), key=lambda x: x["similarity"], reverse=True)

    # ── Step 5: MMR only if we have enough chunks ─────────────────
    if len(merged) >= 3:
        reranked = mmr_rerank(query_combined_emb, merged, lambda_param=0.65, top_k=5)
    else:
        reranked = merged  # too few chunks — skip MMR, take all

    print(f"  After MMR: {len(reranked)} chunks")
    for i, c in enumerate(reranked):
        print(f"  [{i+1}] sim={round(c['similarity'], 3)} src={c['source']} | {c['content'][:100]}...")

    return reranked

STRICT_SYSTEM_TEMPLATE = """You are a strict document assistant. You ONLY answer from the context provided below.

RULES (never break these):
- If the answer is in the context → answer clearly and cite the source filename
- If the answer is NOT in the context → respond EXACTLY: "I don't have information about that in the provided documents."
- NEVER use outside knowledge, assumptions, or training data
- NEVER say things like "Generally speaking..." or "Typically..."
- If context is partially relevant → use only the relevant parts, ignore the rest

CONTEXT:
{context}
---
Answer the user's question using ONLY the above context."""


def build_prompt(system_prompt: str, context_chunks: list[dict], question: str) -> list:
    
    if not context_chunks:
        # Truly no chunks at all
        context_text = "NO DOCUMENTS AVAILABLE."
        confidence   = "none"
    else:
        # Sort by similarity
        context_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        best_score = context_chunks[0]['similarity']

        if best_score > 0.15:
            confidence = "high"
        elif best_score > 0.02:      # ← was 0.25, way too high
            confidence = "medium"
        elif best_score > -0.05:     # ← slightly negative is still usable
            confidence = "low"
        else:
            confidence = "none"

        context_text = "\n\n---\n\n".join([
            f"[Source: {c['filename']} | Match: {round(c['similarity']*100)}% | via {c['source']}]\n{c['content']}"
            for c in context_chunks
        ])

    # Dynamically adjust strictness based on confidence
    if confidence == "none":
        instruction = """No documents were found. 
Say: "I don't have any documents to answer this question." """

    elif confidence == "low":
        instruction = """The retrieved content is not relevant enough to answer the question.
Do NOT attempt to answer.
Respond EXACTLY: "I don't have enough relevant information in the provided documents." """

    elif confidence == "medium":
        instruction = """Answer from the context below. 
If the context partially answers the question, share what is available and note the limitation.
Do NOT invent information not present."""

    else:  # high confidence
        instruction = """Answer directly and thoroughly from the context below.
Cite the source filename when possible."""

    final_system = f"""{system_prompt}

{instruction}

CONTEXT:
{context_text}
---
IMPORTANT: Never use knowledge outside this context. Stick strictly to what is written above."""

    return [
        SystemMessage(content=final_system),
        HumanMessage(content=question),
    ]

async def generate_query_variations(question: str) -> list[str]:
    """Generate alternative phrasings to improve retrieval coverage."""
    prompt = [
        SystemMessage(content="""Generate 3 different ways to ask the same question.
Output ONLY the 3 questions, one per line, no numbering, no explanation."""),
        HumanMessage(content=question)
    ]
    response = await llm.ainvoke(prompt)
    variations = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
    return [question] + variations[:2]  # original + 2 variations = 3 total

    
# MMR (Maximal Marginal Relevance): Instead of returning the top 5 most similar chunks (which are often near-duplicates),
# MMR picks chunks that are both relevant AND different from each other
def mmr_rerank(
    query_embedding: list[float],
    chunks: list[dict],
    lambda_param: float = 0.6,  # 0=max diversity, 1=max relevance
    top_k: int = 6,
) -> list[dict]:
    """
    MMR reranking — picks chunks that are relevant but not too similar to each other.
    Prevents returning 5 near-identical chunks.
    """
    if not chunks:
        return []

    query_vec = np.array(query_embedding)
    chunk_vecs = np.array([c["embedding"] for c in chunks])

    selected = []
    remaining = list(range(len(chunks)))

    for _ in range(min(top_k, len(chunks))):
        if not remaining:
            break

        # Score = relevance to query - similarity to already selected chunks
        scores = []
        for i in remaining:
            relevance = np.dot(query_vec, chunk_vecs[i]) / (
                np.linalg.norm(query_vec) * np.linalg.norm(chunk_vecs[i]) + 1e-8
            )
            if selected:
                redundancy = max(
                    np.dot(chunk_vecs[i], chunk_vecs[j])
                    / (
                        np.linalg.norm(chunk_vecs[i]) * np.linalg.norm(chunk_vecs[j])
                        + 1e-8
                    )
                    for j in selected
                )
            else:
                redundancy = 0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            scores.append((i, mmr_score))

        best_idx = max(scores, key=lambda x: x[1])[0]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [chunks[i] for i in selected]


async def stream_chat_response(
    question: str,
    chatbot_id: str,
    tenant_id: str,
    system_prompt: str,
    db: AsyncSession,
) -> AsyncGenerator[str, None]:

    chunks = await retrieve_relevant_chunks(question, chatbot_id, tenant_id, db)

    # Log what was retrieved — helps debugging in terminal
    print(f"\n🔍 Retrieved {len(chunks)} chunks for: '{question}'")
    for i, c in enumerate(chunks):
        print(
            f"  [{i+1}] {c['filename']} | sim={round(c['similarity'],3)} | src={c['source']} | {c['content'][:80]}..."
        )

    messages = build_prompt(system_prompt, chunks, question)

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content
