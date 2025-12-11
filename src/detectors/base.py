from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseDetector(ABC):
    """
    Abstract base class for all pattern detectors.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @property
    @abstractmethod
    def pattern_name(self) -> str:
        """String identifier for the pattern (e.g., 'double_top')."""
        raise NotImplementedError

    @abstractmethod
    def detect(self, price_series: pd.Series, **kwargs: Any) -> bool:
        """
        Determine whether the pattern exists in the given price window.
        """
        raise NotImplementedError

