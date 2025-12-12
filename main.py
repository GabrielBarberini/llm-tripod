import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from core.config import TripodConfig
from core.prompting import PromptLeg
from core.rag import RAGLeg
from core.training import TrainingLeg

# Configure logging early for consistent output across modules.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Tripod")

DEFAULT_CONFIG_PATH = Path("configs/iot_domain_config.yaml")


class TripodOrchestrator:
    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH):
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)

        # Initialize legs
        self.trainer = TrainingLeg(self.config.training)
        self.rag = RAGLeg(self.config.rag)
        self.prompter = PromptLeg(self.config.prompting)

    def _load_config(self, path: Path) -> TripodConfig:
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path}")

        with path.open("r") as f:
            raw_cfg = yaml.safe_load(f)
        return TripodConfig(**raw_cfg)

    def execute(self, mode: str = "inference", input_payload: Optional[Dict[str, Any]] = None):
        """
        Main entry point for the framework.
        modes: 'train', 'ingest', 'inference', 'evaluate'
        """
        mode = mode.lower()
        logger.info("Tripod executing workflow: %s", mode.upper())

        if mode == "train":
            self.trainer.run()

        elif mode == "ingest":
            documents = (input_payload or {}).get("documents", [])
            self.rag.ingest(documents)

        elif mode == "inference":
            if not input_payload:
                raise ValueError("Input payload required for inference")

            sensor_data = input_payload.get("sensor_data")
            if sensor_data is None:
                raise ValueError("sensor_data is required inside input_payload for inference")

            # Step 1: Retrieve context
            retrieved_context = self.rag.run(query=str(sensor_data))

            # Step 2: Build prompt
            prompt_context = {
                "domain": input_payload.get("domain", "Thermal Control"),
                "rag_context": retrieved_context,
                "sensor_data": sensor_data,
            }
            final_prompt = self.prompter.run(prompt_context)

            # Step 3: Model Generation (placeholder for actual LLM call)
            logger.info("Final prompt ready for LLM engine.")
            print(f"\n--- GENERATED PROMPT ---\n{final_prompt}\n------------------------\n")

        elif mode == "evaluate":
            self.evaluate_system()

        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def evaluate_system(self):
        """
        Runs the full loop against the test set defined in config.
        """
        test_path = self.config.evaluation.test_set_path
        logger.info("Loading test set from %s", test_path)
        logger.info("Running evaluation loop (placeholder).")
        # Loop through test set -> Inference -> Compare with Ground Truth -> Calculate Metrics


if __name__ == "__main__":
    tripod = TripodOrchestrator(DEFAULT_CONFIG_PATH)

    # Example inference payload
    dummy_sensor = {"temp": 78.5, "vibration": 1.2, "status": "warning"}
    tripod.execute("inference", input_payload={"sensor_data": dummy_sensor, "domain": "Thermal Control"})
