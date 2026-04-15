# backend/app/ai/embeddings.py
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import settings

# Singleton — created once, reused for all embedding calls
# embeddings_model = OllamaEmbeddings(
#     model=settings.EMBEDDING_MODEL,       # nomic-embed-text
#     base_url=settings.OLLAMA_BASE_URL,    # http://localhost:11434
# )

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

async def embed_text(text: str) -> list[float]:
    """Convert a string into a vector of numbers."""
    return embeddings_model.embed_query(text)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple strings into vectors (batched for efficiency)."""
    return embeddings_model.embed_documents(texts)