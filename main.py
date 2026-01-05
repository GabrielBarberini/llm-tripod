from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from core.config import TripodConfig
from core.prompting import PromptLeg
from core.rag import RAGLeg
from core.training import TrainingLeg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Tripod")

DEFAULT_CONFIG_PATH = Path("configs/iot_domain_config.yaml")


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
        modes: 'train', 'ingest', 'inference', 'evaluate'
        """
        mode = mode.lower()
        logger.info("Tripod executing workflow: %s", mode.upper())

        match mode:
            case "train":
                self.trainer.run()
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
                match input_payload:
                    case {
                        "sensor_data": sensor_data,
                        **rest,
                    } if sensor_data is not None:
                        domain = rest.get("domain", "Thermal Control")
                    case _:
                        raise ValueError(
                            "Input payload with sensor_data is required for inference."
                        )

                retrieved_context = self.rag.run(query=str(sensor_data))

                prompt_context = {
                    "domain": domain,
                    "rag_context": retrieved_context,
                    "sensor_data": sensor_data,
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
                self.evaluate_stub()
            case _:
                raise ValueError(f"Unsupported mode: {mode}")

    def evaluate_stub(self):
        """
        Stub for evaluation loops against the test set defined in config.
        """
        test_path = self.config.evaluation.test_set_path
        logger.info("Loading test set from %s", test_path)
        logger.info("Running evaluation loop (stub).")


if __name__ == "__main__":
    tripod = TripodOrchestrator(DEFAULT_CONFIG_PATH)

    dummy_sensor = {"temp": 78.5, "vibration": 1.2, "status": "warning"}
    tripod.execute(
        "inference",
        input_payload={
            "sensor_data": dummy_sensor,
            "domain": "Thermal Control",
        },
    )
