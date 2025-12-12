import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.base import BaseLeg
from core.config import RAGConfig
from core.vectordb import LocalVectorStore, VectorDoc

logger = logging.getLogger(__name__)


class RAGLeg(BaseLeg):
    """
    Leg 2: Handles retrieval-augmented generation orchestration.
    Implement ingestion/search against your vector database of choice.
    """

    def __init__(self, config: RAGConfig):
        super().__init__(config)
        self._store: Optional[LocalVectorStore] = None

    def ingest(self, raw_documents: List[Any]):
        if not self.config.enabled:
            logger.info("RAG leg disabled. Skipping ingestion.")
            return

        logger.info(
            "Ingesting %s documents into %s at %s",
            len(raw_documents),
            self.config.vector_db_type,
            self.config.vector_db_path,
        )
        docs: List[VectorDoc] = []
        for row in raw_documents:
            if isinstance(row, str):
                row = {"id": len(docs), "text": row}
            docs.append(
                VectorDoc(
                    id=str(row.get("id", len(docs))),
                    text=str(row["text"]),
                    metadata={k: v for k, v in row.items() if k not in {"text"}},
                )
            )
        store = LocalVectorStore.build(
            docs=docs,
            embedding_model=self.config.ingestion.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        store.save(self.config.vector_db_path)
        self._store = store

    def _ensure_store(self):
        if self._store is not None:
            return
        path = Path(self.config.vector_db_path)
        if path.exists():
            self._store = LocalVectorStore.load(
                dir_path=self.config.vector_db_path,
                embedding_model=self.config.ingestion.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            )

    def run(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        if not self.config.enabled:
            logger.info("RAG leg disabled. Returning empty context.")
            return ""

        self._ensure_store()
        if self._store is None:
            logger.warning("Vector store not found at %s; returning empty context.", self.config.vector_db_path)
            return ""

        k = self.config.retrieval.get("top_k", 5)
        strategy = self.config.retrieval.get("strategy", "similarity")
        logger.info("Retrieving top %s items via %s strategy...", k, strategy)

        results = self._store.query(query_text=query, top_k=k, filters=filters)
        lines = []
        for idx, (doc, score) in enumerate(results, 1):
            lines.append(f"Context_{idx} (score={score:.3f}): {doc.text}")
        return "\n".join(lines)
