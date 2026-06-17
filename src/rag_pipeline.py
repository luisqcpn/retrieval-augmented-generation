"""
RAG (Retrieval-Augmented Generation) pipeline.

Orchestrates the full RAG loop:
  1. Index governance documents into a vector store at startup.
  2. On each query, retrieve semantically relevant passages.
  3. Inject retrieved context into a structured prompt.
  4. Call the Gemini generation model and return a grounded answer.

Restricting the LLM to retrieved context is the core anti-hallucination
mechanism: the model can only assert what the documents explicitly state.
"""

import logging
import os
from typing import Any

from google import genai

from src.document_loader import load_documents
from src.embeddings import build_vector_store, semantic_search

logger = logging.getLogger(__name__)

GENERATION_MODEL = "gemini-2.5-flash"
_DEFAULT_TOP_K = 3
_MAX_CONTEXT_CHARS = 8_000  # guard against exceeding context window


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline over internal governance documents."""

    def __init__(self, documents_path: str, top_k: int = _DEFAULT_TOP_K):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Export it before running the pipeline."
            )

        self.client = genai.Client(api_key=api_key)
        self.top_k = top_k

        logger.info("Initialising RAG pipeline…")
        documents = load_documents(documents_path)
        self.vector_store = build_vector_store(self.client, documents)
        logger.info("Pipeline ready. %d documents indexed.", len(self.vector_store))

    def query(self, question: str) -> dict[str, Any]:
        """
        Run the full RAG loop for a user question.

        Returns a dict with:
          - answer: the model's grounded response
          - sources: list of retrieved documents and their similarity scores
          - question: the original question
        """
        logger.info("Processing question: %r", question)

        retrieved = semantic_search(
            self.client, question, self.vector_store, top_k=self.top_k
        )
        logger.info("Retrieved %d context chunks", len(retrieved))

        prompt = self._build_prompt(question, retrieved)
        logger.debug("Prompt length: %d chars", len(prompt))

        logger.info("Calling generation model (%s)…", GENERATION_MODEL)
        response = self.client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
        answer = response.text.strip()
        logger.info("Answer generated (%d chars)", len(answer))

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "category": doc["category"],
                    "similarity_score": doc["similarity_score"],
                }
                for doc in retrieved
            ],
        }

    def _build_prompt(
        self, question: str, retrieved_docs: list[dict[str, Any]]
    ) -> str:
        """
        Construct the RAG prompt with injected context.

        The system instruction explicitly forbids the model from using
        knowledge outside the provided documents, which is the primary
        mechanism for hallucination suppression.
        """
        context_parts: list[str] = []
        total_chars = 0

        for doc in retrieved_docs:
            block = (
                f"[Documento: {doc['id']} — {doc['title']} | Categoria: {doc['category']}]\n"
                f"{doc['content']}"
            )
            if total_chars + len(block) > _MAX_CONTEXT_CHARS:
                logger.warning(
                    "Context limit reached after %d chars; truncating remaining docs",
                    total_chars,
                )
                break
            context_parts.append(block)
            total_chars += len(block)

        context_block = "\n\n---\n\n".join(context_parts)

        return f"""Você é um assistente especializado em Governança de Dados e LGPD de uma empresa brasileira.
Responda à pergunta do usuário EXCLUSIVAMENTE com base nos documentos internos fornecidos abaixo.

REGRAS OBRIGATÓRIAS:
1. Use APENAS as informações presentes nos documentos fornecidos.
2. Se a resposta não estiver nos documentos, diga explicitamente: "Esta informação não está disponível nos documentos de governança consultados."
3. Cite os IDs dos documentos de onde extraiu cada informação (ex: [pol-001]).
4. Não invente prazos, nomes de leis, artigos ou obrigações que não estejam nos documentos.
5. Seja preciso, claro e profissional.

=== DOCUMENTOS RECUPERADOS ===

{context_block}

=== PERGUNTA DO USUÁRIO ===

{question}

=== RESPOSTA ==="""
