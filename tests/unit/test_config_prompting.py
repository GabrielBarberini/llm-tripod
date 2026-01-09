from __future__ import annotations

from core.config import TripodConfig


def test_config_parses_prompting_system_prompt(smoke_config: dict):
    """TripodConfig should parse prompting.system_prompt from YAML configs."""

    cfg = TripodConfig.model_validate(smoke_config)
    assert isinstance(cfg.prompting.system_prompt, str)
    assert cfg.prompting.system_prompt.strip()
