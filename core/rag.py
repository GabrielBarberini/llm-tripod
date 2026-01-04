from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
        self._store: LocalVectorStore | None = None

    def ingest(self, raw_documents: list[Any]) -> None:
        match self.config.enabled:
            case False:
                logger.info("RAG leg disabled. Skipping ingestion.")
                return
            case True:
                pass

        logger.info(
            "Ingesting %s documents into %s at %s",
            len(raw_documents),
            self.config.vector_db_type,
            self.config.vector_db_path,
        )
        docs: list[VectorDoc] = []
        for idx, row in enumerate(raw_documents):
            match row:
                case str() as text:
                    docs.append(VectorDoc(id=str(idx), text=text, metadata={}))
                case {"text": str() as text, **rest}:
                    doc_id = rest.get("id", idx)
                    metadata = dict(rest)
                    docs.append(
                        VectorDoc(
                            id=str(doc_id),
                            text=text,
                            metadata=metadata,
                        )
                    )
                case _:
                    logger.warning(
                        "Skipping unsupported RAG document type: %s",
                        type(row),
                    )

        if not docs:
            logger.warning("No valid RAG documents to ingest.")
            return

        store = LocalVectorStore.build(
            docs=docs,
            embedding_model=self._embedding_model(),
        )
        store.save(self.config.vector_db_path)
        self._store = store

    def _ensure_store(self) -> None:
        if self._store is not None:
            return
        path = Path(self.config.vector_db_path)
        match path.exists():
            case True:
                self._store = LocalVectorStore.load(
                    dir_path=self.config.vector_db_path,
                    embedding_model=self._embedding_model(),
                )
            case False:
                return

    def _embedding_model(self) -> str:
        match self.config.ingestion:
            case {"embedding_model": str() as embedding_model}:
                return embedding_model
            case _:
                return "sentence-transformers/all-MiniLM-L6-v2"

    def _retrieval_settings(self) -> tuple[int, str]:
        match self.config.retrieval:
            case {"top_k": int() as top_k, "strategy": str() as strategy}:
                return top_k, strategy
            case {"top_k": int() as top_k}:
                return top_k, "similarity"
            case {"strategy": str() as strategy}:
                return 5, strategy
            case _:
                return 5, "similarity"

    def run(self, query: str, filters: dict[str, Any] | None = None) -> str:
        match self.config.enabled:
            case False:
                logger.info("RAG leg disabled. Returning empty context.")
                return ""
            case True:
                pass

        self._ensure_store()
        if self._store is None:
            logger.warning(
                "Vector store not found at %s; returning empty context.",
                self.config.vector_db_path,
            )
            return ""

        top_k, strategy = self._retrieval_settings()
        logger.info(
            "Retrieving top %s items via %s strategy...", top_k, strategy
        )

        match filters:
            case None:
                resolved_filters = None
            case dict() as filter_map:
                resolved_filters = filter_map
            case _:
                raise ValueError("filters must be a dict or None")

        results = self._store.query(
            query_text=query, top_k=top_k, filters=resolved_filters
        )
        lines: list[str] = []
        for idx, (doc, score) in enumerate(results, 1):
            lines.append(f"Context_{idx} (score={score:.3f}): {doc.text}")
        return "\n".join(lines)
