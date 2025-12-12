from abc import ABC, abstractmethod
from typing import Any


class BaseLeg(ABC):
    """Abstract base class for each Tripod leg."""

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        """Execute the leg's primary operation."""
        raise NotImplementedError
