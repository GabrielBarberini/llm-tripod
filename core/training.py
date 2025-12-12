import logging
from typing import Any

from core.base import BaseLeg
from core.config import TrainingConfig

logger = logging.getLogger(__name__)


class TrainingLeg(BaseLeg):
    """
    Leg 1: Handles model fine-tuning (LoRA/PEFT).
    Integrate your trainer of choice inside `run`.
    """

    def __init__(self, config: TrainingConfig):
        super().__init__(config)

    def run(self, input_data: Any = None):
        if not self.config.enabled:
            logger.info("Training leg disabled. Skipping.")
            return

        logger.info("Starting training on base model: %s", self.config.base_model)
        logger.info("Dataset path: %s", self.config.dataset_path)
        logger.info("LoRA config: %s", self.config.lora_config.dict())
        logger.info("Adapter output dir: %s", self.config.adapter_output_dir)
        logger.info("Hyperparameters: %s", self.config.hyperparameters)

        # --- PRODUCTION LOGIC PLACEHOLDER ---
        # 1. Load/quantize the base model.
        # 2. Stream or load the training dataset.
        # 3. Apply LoRA/PEFT configuration.
        # 4. Run trainer.train() with gradient accumulation.
        # 5. Persist adapter weights to adapter_output_dir.
        # ------------------------------------

        logger.info("Training complete. Adapter saved to %s", self.config.adapter_output_dir)
