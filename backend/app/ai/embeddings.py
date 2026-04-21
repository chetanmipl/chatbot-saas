# backend/app/ai/embeddings.py
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

# Singleton — created once, reused for all embedding calls
# embeddings_model = OllamaEmbeddings(
#     model=settings.EMBEDDING_MODEL,  # nomic-embed-text
#     base_url=settings.OLLAMA_BASE_URL,  # http://localhost:11434
# )

# Alternative: HuggingFace embeddings (requires: pip install langchain-huggingface)
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )


from sentence_transformers import SentenceTransformer
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
embeddings_model=SentenceTransformer("BAAI/bge-base-en")

import numpy as np

def normalize_embedding(embedding: list[float]) -> list[float]:
    vec  = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return embedding
    return (vec / norm).tolist()

#used with huggingface
# async def embed_text(text: str) -> list[float]:
#     raw = embeddings_model.embed_query(text)
#     return normalize_embedding(raw)

# async def embed_texts(texts: list[str]) -> list[list[float]]:
#     raw = embeddings_model.embed_documents(texts)
#     return [normalize_embedding(e) for e in raw]

async def embed_text(text: str, is_query: bool = True) -> list[float]:
    """
    BGE requires a prefix on QUERIES but NOT on documents.
    is_query=True  → searching (add prefix)
    is_query=False → storing document chunks (no prefix)
    """
    if is_query:
        text = BGE_QUERY_PREFIX + text
    raw = embeddings_model.encode(text, normalize_embeddings=True).tolist()
    return raw


async def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """For document chunks — no prefix."""
    raw = embeddings_model.encode(texts, normalize_embeddings=True).tolist()
    return raw


# async def embed_text(text: str) -> list[float]:
#     """Convert a string into a vector of numbers."""
#     return embeddings_model.embed_query(text)


# async def embed_texts(texts: list[str]) -> list[list[float]]:
#     """Convert multiple strings into vectors (batched for efficiency)."""
#     return embeddings_model.embed_documents(texts)
