from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.config import TripodConfig
from core.evaluation import evaluator

logger = logging.getLogger(__name__)


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("Test set not found: %s", path)
        return []

    match path.suffix.lower():
        case ".jsonl":
            rows: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        match json.loads(line):
                            case dict() as row:
                                rows.append(row)
                            case _:
                                raise ValueError(
                                    "Each JSONL row must be an object."
                                )
            return rows
        case ".json":
            match json.loads(path.read_text(encoding="utf-8")):
                case list() as rows:
                    for row in rows:
                        match row:
                            case dict():
                                continue
                            case _:
                                raise ValueError(
                                    "Each JSON row must be an object."
                                )
                    return rows
                case dict() as row:
                    return [row]
                case _:
                    raise ValueError("JSON dataset must be an object list.")
        case _:
            logger.warning("Unsupported test set format: %s", path.suffix)
            return []


@evaluator("iot")
def evaluate_iot(
    config: TripodConfig, input_payload: dict[str, Any] | None = None
) -> dict[str, Any]:
    match input_payload:
        case {"test_set_path": str() as path, **_rest}:
            test_path = Path(path)
        case _:
            test_path = Path(config.evaluation.test_set_path)

    rows = _load_json_rows(test_path)
    total = len(rows)
    if total == 0:
        metrics = {"sample_count": 0, "schema_validity": 0.0}
        logger.info("IoT evaluation metrics: %s", metrics)
        return metrics

    valid = 0
    for row in rows:
        match row:
            case {"expected": {"action": str(), "parameters": dict()}}:
                valid += 1
            case {"target": {"action": str(), "parameters": dict()}}:
                valid += 1
            case _:
                pass

    metrics = {"sample_count": total, "schema_validity": valid / total}
    logger.info("IoT evaluation metrics: %s", metrics)
    return metrics
