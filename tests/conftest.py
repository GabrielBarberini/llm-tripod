"""Shared pytest fixtures for Tripod tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def smoke_config_path() -> Path:
    """Return path to smoke e2e config file."""

    return ROOT / "configs" / "smoke_e2e_config.yaml"


@pytest.fixture()
def smoke_config(smoke_config_path: Path) -> dict[str, Any]:
    """Return parsed smoke e2e config as dict."""

    with smoke_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def iot_config_path() -> Path:
    """Return path to IoT domain config file."""

    return ROOT / "configs" / "iot_domain_config.yaml"


@pytest.fixture()
def iot_config(iot_config_path: Path) -> dict[str, Any]:
    """Return parsed IoT domain config as dict."""

    with iot_config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def temp_training_data_dir(tmp_path: Path) -> Path:
    """Return a temporary directory for training data artifacts."""

    training_data = tmp_path / "training_data"
    training_data.mkdir(parents=True, exist_ok=True)
    return training_data
