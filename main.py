from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from core.config import TripodConfig
from core.evaluation import get_evaluator, load_entrypoint
from core.prompting import PromptLeg
from core.rag import RAGLeg
from core.training import TrainingLeg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Tripod")

DEFAULT_CONFIG_PATH = Path("configs/iot_config.yaml")


class TripodOrchestrator:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        self.trainer = TrainingLeg(self.config.training)
        self.raft = RAGLeg(self.config.raft)
        self.rag = RAGLeg(self.config.rag)
        self.prompter = PromptLeg(self.config.prompting)

    def _load_config(self, path: Path) -> TripodConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path}")

        raw_cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        match raw_cfg:
            case dict() as payload:
                return TripodConfig.model_validate(payload)
            case _:
                raise ValueError("Config root must be a mapping.")

    def _select_retrieval(self, target: str) -> RAGLeg:
        match target.lower():
            case "raft" | "training":
                return self.raft
            case "rag" | "inference":
                return self.rag
            case _:
                raise ValueError(f"Unsupported retrieval target: {target}")

    def execute(
        self,
        mode: str = "inference",
        input_payload: dict[str, Any] | None = None,
    ):
        """
        Main entry point for the framework.
        modes: 'prepare_train', 'train', 'ingest', 'inference', 'evaluate'
        """
        mode = mode.lower()
        logger.info("Tripod executing workflow: %s", mode.upper())

        match mode:
            case "train":
                self.trainer.run()
            case "prepare_train":
                match input_payload:
                    case {"dataset_path": str() as dataset_path, **_rest}:
                        self.config.training.dataset_path = dataset_path
                    case None | {}:
                        pass
                    case _:
                        raise ValueError(
                            "Input payload for prepare_train must be empty or include dataset_path."
                        )
                prepared_path = self._prepare_training_dataset()
                if prepared_path is not None:
                    self.config.training.dataset_path = str(prepared_path)
                else:
                    logger.info(
                        "Training dataset already SFT or not a local JSON/JSONL; skipping preparation."
                    )
            case "ingest":
                match input_payload:
                    case {
                        "documents": list() as documents,
                        "target": str() as target,
                        **_rest,
                    }:
                        retrieval_target = target
                    case {"documents": list() as documents, **_rest}:
                        retrieval_target = "rag"
                    case _:
                        raise ValueError(
                            "Input payload with documents list is required for ingest."
                        )
                self._select_retrieval(retrieval_target).ingest(documents)
            case "inference":
                match input_payload.get("input_data"):
                    case None:
                        raise ValueError(
                            "Input payload with 'input_data' is required for inference."
                        )
                    case task_input:
                        task_label = input_payload.get(
                            "task_label", "Thermal Control"
                        )

                retrieved_context = self.rag.run(query=str(task_input))

                prompt_context = {
                    "task_label": task_label,
                    "rag_context": retrieved_context,
                    "input_data": task_input,
                }
                prompt_output = self.prompter.run(prompt_context)

                match self.prompter.backend:
                    case "dspy":
                        logger.info("DSPy output ready.")
                        print(
                            "\n--- DSPY OUTPUT ---\n"
                            f"{prompt_output}\n"
                            "-------------------\n"
                        )
                    case _:
                        logger.info("Final prompt ready for LLM engine.")
                        print(
                            "\n--- GENERATED PROMPT ---\n"
                            f"{prompt_output}\n"
                            "------------------------\n"
                        )
            case "evaluate":
                self.evaluate(input_payload)
            case _:
                raise ValueError(f"Unsupported mode: {mode}")

    def _prepare_training_dataset(self) -> Path | None:
        dataset_path = Path(self.config.training.dataset_path)
        if not dataset_path.exists():
            return None

        match dataset_path.suffix.lower():
            case ".jsonl" | ".json":
                pass
            case _:
                return None

        first_row = self._peek_json_row(dataset_path)
        match first_row:
            case {"text": str()}:
                return None
            case None:
                return None
            case _:
                pass

        rows = self._load_json_rows(dataset_path)
        response_marker = str(
            self.config.training.hyperparameters.get(
                "response_marker", "\nASSISTANT:\n"
            )
        )
        use_raft = bool(self.raft.config.enabled)
        suffix = "raft" if use_raft else "no_raft"
        output_path = dataset_path.with_name(
            f"{dataset_path.stem}_{suffix}_sft.jsonl"
        )

        sft_rows = []
        for row in rows:
            task_label = row.get("task_label", "IoT")
            match row.get("input_data"):
                case dict() | list() as data:
                    resolved_input = data
                case None:
                    resolved_input = {}
                case other:
                    resolved_input = other

            rag_context = self._resolve_raft_context(row)
            prompt = self.prompter.render_prompt(
                {
                    "task_label": task_label,
                    "rag_context": rag_context,
                    "input_data": resolved_input,
                }
            )
            target = self._resolve_target(row)
            sft_rows.append({"text": f"{prompt}{response_marker}{target}"})

        self._write_jsonl(output_path, sft_rows)
        logger.info("Prepared SFT dataset at %s", output_path)
        return output_path

    def _resolve_raft_context(self, row: dict[str, Any]) -> str:
        match row.get("rag_context"):
            case str() as rag_context if rag_context.strip():
                return rag_context
            case _:
                pass

        if not self.raft.config.enabled:
            return ""

        query = self._resolve_raft_query(row)
        if not query:
            return ""

        filters = self._resolve_raft_filters(row)
        return self.raft.run(query=query, filters=filters)

    def _resolve_raft_query(self, row: dict[str, Any]) -> str:
        match row.get("raft_query"):
            case str() as query if query.strip():
                return query.strip()
            case _:
                pass

        match row.get("rag_query"):
            case str() as query if query.strip():
                return query.strip()
            case _:
                pass

        match row.get("input_data"):
            case dict() | list() as data:
                return json.dumps(data, ensure_ascii=False, sort_keys=True)
            case str() as query if query.strip():
                return query.strip()
            case other if other is not None:
                return str(other)
            case _:
                return ""

    def _resolve_raft_filters(
        self, row: dict[str, Any]
    ) -> dict[str, Any] | None:
        match row.get("raft_filters"):
            case dict() as filters:
                return filters
            case _:
                pass

        match row.get("rag_filters"):
            case dict() as filters:
                return filters
            case _:
                return None

    def _resolve_target(self, row: dict[str, Any]) -> str:
        match row:
            case {"target": str() as target} if target.strip():
                return target
            case {"expected": str() as expected} if expected.strip():
                return expected
            case {"expected": dict() as expected}:
                return json.dumps(expected, ensure_ascii=False)
            case {"target": dict() as target}:
                return json.dumps(target, ensure_ascii=False)
            case _:
                raise ValueError(
                    "Training rows must include 'expected' or 'target'."
                )

    def _peek_json_row(self, path: Path) -> dict[str, Any] | None:
        match path.suffix.lower():
            case ".jsonl":
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            match json.loads(line):
                                case dict() as row:
                                    return row
                                case _:
                                    return None
                return None
            case ".json":
                match json.loads(path.read_text(encoding="utf-8")):
                    case list() as rows if rows:
                        match rows[0]:
                            case dict() as row:
                                return row
                            case _:
                                return None
                    case dict() as row:
                        return row
                    case _:
                        return None
            case _:
                return None

    def _load_json_rows(self, path: Path) -> list[dict[str, Any]]:
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
                        raise ValueError(
                            "JSON dataset must be an object list."
                        )
            case _:
                raise ValueError(
                    "Training dataset must be a JSON or JSONL file."
                )

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def evaluate(self, input_payload: dict[str, Any] | None = None):
        eval_cfg = self.config.evaluation
        match eval_cfg.entrypoint:
            case str() as entrypoint if entrypoint.strip():
                load_entrypoint(entrypoint)
            case _:
                pass

        evaluator_name = eval_cfg.evaluator or "stub"
        evaluator = get_evaluator(evaluator_name)
        if evaluator is None and evaluator_name != "stub":
            logger.info(
                "Evaluator '%s' not found; falling back to stub.",
                evaluator_name,
            )
            evaluator = get_evaluator("stub")

        if evaluator is None:
            logger.info("No evaluator registered; skipping evaluation.")
            return None

        return evaluator(self.config, input_payload)


if __name__ == "__main__":
    tripod = TripodOrchestrator(DEFAULT_CONFIG_PATH)

    dummy_input = {"temp": 78.5, "vibration": 1.2, "status": "warning"}
    tripod.execute(
        "inference",
        input_payload={
            "input_data": dummy_input,
            "task_label": "Thermal Control",
        },
    )
