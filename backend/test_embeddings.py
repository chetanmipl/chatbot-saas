# Run this once in a Python shell to calibrate your model
# backend/test_embedding.py

import asyncio
from app.ai.embeddings import embed_text
import numpy as np

async def test():
    a = await embed_text("Served as a Volunteer for the Valorant Tournament")
    b = await embed_text("games and gaming events")
    c = await embed_text("cooking recipes")

    def cosine(x, y):
        x, y = np.array(x), np.array(y)
        return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    print(f"Valorant ↔ games:   {cosine(a, b):.4f}")  # expect ~0.3-0.5
    print(f"Valorant ↔ cooking: {cosine(a, c):.4f}")  # expect ~0.0-0.1

asyncio.run(test())