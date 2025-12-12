import logging
from typing import Any, Dict

from core.base import BaseLeg
from core.config import PromptingConfig

logger = logging.getLogger(__name__)


class PromptLeg(BaseLeg):
    """
    Leg 3: Handles dynamic prompt construction with simple template substitution.
    Swap in a templating engine (e.g., Jinja2) if you need richer rendering.
    """

    def __init__(self, config: PromptingConfig):
        super().__init__(config)

    def run(self, context: Dict[str, Any]) -> str:
        logger.info("Constructing prompt using template: %s", self.config.template_id)

        system_prompt = self.config.system_prompt.replace("{{ domain }}", context.get("domain", "IoT"))
        user_prompt = self.config.user_prompt_structure.replace("{{ rag_context }}", context.get("rag_context", ""))
        user_prompt = user_prompt.replace("{{ sensor_data }}", str(context.get("sensor_data", {})))

        full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        return full_prompt
