import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_documents(filepath: str | Path) -> list[dict[str, Any]]:
    """Load governance documents from a JSON file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Document file not found: {path}")

    logger.info("Loading documents from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    logger.info("Loaded %d documents", len(documents))
    for doc in documents:
        _validate_document(doc)

    return documents


def _validate_document(doc: dict[str, Any]) -> None:
    required_fields = {"id", "title", "category", "content"}
    missing = required_fields - set(doc.keys())
    if missing:
        raise ValueError(f"Document {doc.get('id', '?')} missing fields: {missing}")
