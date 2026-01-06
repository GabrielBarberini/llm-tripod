from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

from core.config import TripodConfig

logger = logging.getLogger(__name__)

Evaluator = Callable[
    [TripodConfig, dict[str, Any] | None], dict[str, Any] | None
]

_EVALUATORS: dict[str, Evaluator] = {}


def register_evaluator(name: str, evaluator: Evaluator) -> None:
    match name:
        case str() as key if key.strip():
            _EVALUATORS[key] = evaluator
        case _:
            raise ValueError("Evaluator name must be a non-empty string.")


def evaluator(name: str):
    def _decorator(fn: Evaluator) -> Evaluator:
        register_evaluator(name, fn)
        return fn

    return _decorator


def get_evaluator(name: str) -> Evaluator | None:
    return _EVALUATORS.get(name)


def load_entrypoint(entrypoint: str) -> None:
    match entrypoint:
        case str() as path if path.strip():
            importlib.import_module(path)
        case _:
            return None


def stub_evaluator(
    config: TripodConfig, input_payload: dict[str, Any] | None = None
) -> None:
    _ = input_payload
    logger.info(
        "Evaluation stub: test set is %s", config.evaluation.test_set_path
    )
    logger.info("Provide a pipeline-specific evaluator to compute metrics.")


register_evaluator("stub", stub_evaluator)
