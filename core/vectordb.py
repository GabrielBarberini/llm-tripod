from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VectorDoc:
    id: str
    text: str
    metadata: dict[str, Any]


class LocalVectorStore:
    """
    Minimal local vector store backed by numpy arrays + a JSONL doc file.
    Designed for smoke tests and small-scale local RAG.
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        docs: list[VectorDoc],
        embeddings: np.ndarray,
    ):
        self.embedder = embedder
        self.docs = docs
        self.embeddings = embeddings  # shape: (n_docs, dim), normalized

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
        return x / denom

    @classmethod
    def build(
        cls,
        docs: list[VectorDoc],
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> "LocalVectorStore":
        embedder = SentenceTransformer(embedding_model, device=device)
        texts = [d.text for d in docs]
        emb = embedder.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )
        emb = cls._normalize(emb.astype(np.float32))
        return cls(embedder=embedder, docs=docs, embeddings=emb)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[VectorDoc, float]]:
        q = self.embedder.encode(
            [query_text], convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)[0]
        q = self._normalize(q)
        scores = self.embeddings @ q

        idxs = np.argsort(scores)[::-1]
        results: list[tuple[VectorDoc, float]] = []
        match filters:
            case None:
                resolved_filters = None
            case dict() as filter_map:
                resolved_filters = filter_map
            case _:
                raise ValueError("filters must be a dict or None")
        for i in idxs:
            doc = self.docs[int(i)]
            if resolved_filters:
                ok = True
                for k, v in resolved_filters.items():
                    if doc.metadata.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue
            results.append((doc, float(scores[int(i)])))
            if len(results) >= top_k:
                break
        return results

    @classmethod
    def load(
        cls,
        dir_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        p = Path(dir_path)
        docs_path = p / "docs.jsonl"
        emb_path = p / "embeddings.npy"
        if not docs_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                f"Vector store missing files in {p} (need docs.jsonl and embeddings.npy)"
            )

        docs: list[VectorDoc] = []
        with docs_path.open("r", encoding="utf-8") as f:
            for line in f:
                match json.loads(line):
                    case {
                        "id": doc_id,
                        "text": str() as text,
                        "metadata": dict() as metadata,
                    }:
                        docs.append(
                            VectorDoc(
                                id=str(doc_id),
                                text=text,
                                metadata=metadata,
                            )
                        )
                    case {"id": doc_id, "text": str() as text}:
                        docs.append(
                            VectorDoc(
                                id=str(doc_id),
                                text=text,
                                metadata={},
                            )
                        )
                    case _:
                        continue

        embeddings = np.load(str(emb_path)).astype(np.float32)
        embedder = SentenceTransformer(embedding_model, device=device)
        return cls(embedder=embedder, docs=docs, embeddings=embeddings)

    def save(self, dir_path: str):
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        docs_path = p / "docs.jsonl"
        emb_path = p / "embeddings.npy"

        with docs_path.open("w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(
                    json.dumps(
                        {"id": d.id, "text": d.text, "metadata": d.metadata}
                    )
                    + "\n"
                )
        np.save(str(emb_path), self.embeddings)
        logger.info("Saved vector store to %s", p)
