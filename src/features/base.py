from abc import ABC, abstractmethod
from typing import Any, Dict, List

import pandas as pd


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for computing features from a price window.
    """

    @abstractmethod
    def compute_features(self, close_window: pd.Series, **kwargs: Any) -> Dict[str, float]:
        """Return a dictionary of engineered features for a window."""
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """List of feature names in deterministic order."""
        raise NotImplementedError

