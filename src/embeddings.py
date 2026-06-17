"""
Semantic embedding generation and cosine-similarity vector search.

Uses Google's text-embedding-004 model to convert text chunks into
high-dimensional vectors, then ranks documents by cosine similarity
against a query vector to retrieve the most semantically relevant passages.
"""

import logging
import time
from typing import Any

import numpy as np
from google import genai

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "gemini-embedding-001"
_RATE_LIMIT_DELAY = 0.5  # seconds between API calls to avoid 429s


def generate_embedding(client: genai.Client, text: str) -> list[float]:
    """Return the embedding vector for a single text string."""
    logger.debug("Generating embedding for text (len=%d chars)", len(text))
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def build_vector_store(
    client: genai.Client, documents: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Embed every document and attach its vector to the record.

    Returns a list of dicts with the original document fields plus an
    'embedding' key holding a numpy array of shape (D,).
    """
    logger.info("Building vector store for %d documents…", len(documents))
    vector_store: list[dict[str, Any]] = []

    for i, doc in enumerate(documents, start=1):
        # Combine title and content so retrieval benefits from structural context
        text_to_embed = f"{doc['title']}\n\n{doc['content']}"
        logger.info("  [%d/%d] Embedding document '%s'", i, len(documents), doc["id"])

        embedding = generate_embedding(client, text_to_embed)
        vector_store.append(
            {
                **doc,
                "embedding": np.array(embedding, dtype=np.float32),
            }
        )
        # Respect API rate limits during bulk indexing
        if i < len(documents):
            time.sleep(_RATE_LIMIT_DELAY)

    logger.info("Vector store built: %d entries indexed", len(vector_store))
    return vector_store


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalised vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def semantic_search(
    client: genai.Client,
    query: str,
    vector_store: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-k most relevant documents for the given query.

    Pipeline:
      1. Embed the user query with the same model used for indexing.
      2. Compute cosine similarity against every stored document vector.
      3. Return the top-k documents sorted by descending similarity score.
    """
    logger.info("Running semantic search (query len=%d chars, top_k=%d)", len(query), top_k)
    query_embedding = np.array(
        generate_embedding(client, query), dtype=np.float32
    )

    scored: list[tuple[float, dict[str, Any]]] = []
    for entry in vector_store:
        score = cosine_similarity(query_embedding, entry["embedding"])
        scored.append((score, entry))
        logger.debug("  doc=%s  similarity=%.4f", entry["id"], score)

    scored.sort(key=lambda x: x[0], reverse=True)
    top_results = scored[:top_k]

    logger.info(
        "Top-%d results: %s",
        top_k,
        [(doc["id"], f"{score:.4f}") for score, doc in top_results],
    )
    return [
        {**doc, "similarity_score": round(score, 4)}
        for score, doc in top_results
    ]
