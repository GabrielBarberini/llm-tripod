from __future__ import annotations

from core.config import PromptingConfig
from core.prompting import PromptLeg


def test_prompting_renders_task_label_and_domain_compat():
    """PromptLeg should render task_label into the system prompt template."""

    cfg = PromptingConfig.model_validate(
        {
            "template_id": "t1",
            "system_prompt": "Hello {{ task_label }}",
            "user_prompt_structure": "Input={{ input_data }}",
            "backend": "raw",
        }
    )
    leg = PromptLeg(cfg)

    prompt = leg.render_prompt(
        {"task_label": "Thermal Control", "input_data": {}}
    )
    assert "Hello Thermal Control" in prompt
