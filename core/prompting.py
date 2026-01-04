from __future__ import annotations

import json
import logging
from typing import Any

from core.base import BaseLeg
from core.config import DSPyConfig, PromptingConfig

logger = logging.getLogger(__name__)


class PromptLeg(BaseLeg):
    """
    Leg 3: Handles dynamic prompt construction with DSPy-backed or template-based prompting.
    """

    def __init__(self, config: PromptingConfig):
        super().__init__(config)
        backend = getattr(config, "backend", "raw") or "raw"
        self._backend = str(backend).lower()
        self._dspy_module = None
        self._dspy_output_field: str | None = None
        self._dspy = None

    @property
    def backend(self) -> str:
        return self._backend

    def run(self, context: dict[str, Any]) -> str:
        match self._backend:
            case "raw":
                logger.info(
                    "Constructing prompt using template: %s",
                    self.config.template_id,
                )
                return self.render_prompt(context)
            case "dspy":
                logger.info(
                    "Running DSPy program for template: %s",
                    self.config.template_id,
                )
                return self.predict(context)
            case _:
                raise ValueError(
                    f"Unsupported prompting backend: {self._backend}"
                )

    def render_prompt(self, context: dict[str, Any]) -> str:
        domain, rag_context, sensor_data_str = self._normalize_context(context)

        system_prompt = self.config.system_prompt.replace(
            "{{ domain }}", domain
        )
        user_prompt = self.config.user_prompt_structure.replace(
            "{{ rag_context }}", rag_context
        )
        user_prompt = user_prompt.replace("{{ sensor_data }}", sensor_data_str)

        full_prompt = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
        return full_prompt

    def predict(self, context: dict[str, Any]) -> str:
        domain, rag_context, sensor_data_str = self._normalize_context(context)
        dspy = self._ensure_dspy()
        if getattr(dspy.settings, "lm", None) is None:
            raise RuntimeError(
                "DSPy backend requires a configured LM. Call dspy.settings.configure(lm=...) before use "
                "or set prompting.backend to 'raw'."
            )
        prediction = self._dspy_module(
            domain=domain, rag_context=rag_context, sensor_data=sensor_data_str
        )
        return self._extract_prediction(prediction, self._dspy_output_field)

    def _normalize_context(
        self, context: dict[str, Any]
    ) -> tuple[str, str, str]:
        match context.get("domain"):
            case str() as domain if domain.strip():
                resolved_domain = domain.strip()
            case _:
                resolved_domain = "IoT"

        match context.get("rag_context"):
            case str() as rag_context:
                resolved_rag = rag_context
            case None:
                resolved_rag = ""
            case other:
                resolved_rag = str(other)

        match context.get("sensor_data"):
            case dict() | list() as sensor_data:
                sensor_data_str = json.dumps(
                    sensor_data, ensure_ascii=False, sort_keys=True
                )
            case None:
                sensor_data_str = "{}"
            case other:
                sensor_data_str = str(other)

        return resolved_domain, resolved_rag, sensor_data_str

    def _ensure_dspy(self):
        if self._dspy_module is not None:
            return self._dspy

        try:
            import dspy  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "DSPy backend requested but dspy is not installed. Install it with `pip install dspy-ai`."
            ) from exc

        dspy_cfg = self.config.dspy or DSPyConfig()
        output_field = dspy_cfg.output_field or "response"
        match output_field:
            case str() as field if not field.isidentifier():
                raise ValueError(
                    "DSPy output_field must be a valid identifier: "
                    f"{output_field}"
                )
            case str() as field if field in {
                "domain",
                "rag_context",
                "sensor_data",
            }:
                raise ValueError(
                    "DSPy output_field conflicts with input field: "
                    f"{output_field}"
                )
            case _:
                pass

        instructions = self._build_dspy_instructions(dspy_cfg)
        signature = self._build_dspy_signature(
            dspy,
            instructions=instructions,
            output_field=output_field,
            output_desc=dspy_cfg.output_desc,
        )

        class TripodDSPyModule(dspy.Module):
            def __init__(self):
                super().__init__()
                if dspy_cfg.chain_of_thought:
                    self.predict = dspy.ChainOfThought(signature)
                else:
                    self.predict = dspy.Predict(signature)

            def forward(self, domain: str, rag_context: str, sensor_data: str):
                return self.predict(
                    domain=domain,
                    rag_context=rag_context,
                    sensor_data=sensor_data,
                )

        self._dspy_module = TripodDSPyModule()
        self._dspy_output_field = output_field
        self._dspy = dspy
        return dspy

    def _build_dspy_instructions(self, dspy_cfg: DSPyConfig) -> str:
        if dspy_cfg.instructions:
            instructions = self._sanitize_template(
                dspy_cfg.instructions
            ).strip()
        else:
            instructions = self._sanitize_template(
                self.config.system_prompt
            ).strip()

        if (
            dspy_cfg.include_user_prompt
            and self.config.user_prompt_structure.strip()
        ):
            user_prompt = self._sanitize_template(
                self.config.user_prompt_structure
            ).strip()
            if instructions:
                instructions = f"{instructions}\n\n{user_prompt}"
            else:
                instructions = user_prompt

        return instructions

    def _sanitize_template(self, template: str) -> str:
        return (
            template.replace("{{ domain }}", "[domain]")
            .replace("{{ rag_context }}", "[rag_context]")
            .replace("{{ sensor_data }}", "[sensor_data]")
        )

    def _build_dspy_signature(
        self, dspy, instructions: str, output_field: str, output_desc: str
    ):
        attrs = {
            "__doc__": instructions,
            "domain": dspy.InputField(desc="Domain for the control task."),
            "rag_context": dspy.InputField(desc="Retrieved context snippets."),
            "sensor_data": dspy.InputField(
                desc="Current sensor readings as JSON."
            ),
            output_field: dspy.OutputField(desc=output_desc),
        }
        return type("TripodPromptSignature", (dspy.Signature,), attrs)

    def _extract_prediction(
        self, prediction: Any, output_field: str | None
    ) -> str:
        match (prediction, output_field):
            case (_, str() as field) if hasattr(prediction, field):
                return str(getattr(prediction, field))
            case (dict() as payload, str() as field) if field in payload:
                return str(payload[field])
            case (str() as text, _):
                return text
            case _:
                return str(prediction)
