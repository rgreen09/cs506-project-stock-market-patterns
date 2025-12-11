from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config.settings import load_config
from src.detectors.cup_and_handle import CupAndHandleDetector
from src.detectors.double_top import DoubleTopDetector
from src.detectors.flags_pennants import FlagsPennantsDetector
from src.detectors.head_and_shoulders import HeadAndShouldersDetector
from src.detectors.triangle import TriangleDetector
from src.features.cup_and_handle import CupAndHandleFeatures
from src.features.double_top import DoubleTopFeatures
from src.features.flags_pennants import FlagsPennantsFeatures
from src.features.head_and_shoulders import HeadAndShouldersFeatures
from src.features.triangle import TriangleFeatures
from src.utils.io import append_rows, resolve_output_path, write_header
from src.utils.time_filters import is_within_trading_hours


DETECTOR_REGISTRY = {
    "double_top": (DoubleTopDetector, DoubleTopFeatures),
    "flags_pennants": (FlagsPennantsDetector, FlagsPennantsFeatures),
    "head_and_shoulders": (HeadAndShouldersDetector, HeadAndShouldersFeatures),
    "triangle": (TriangleDetector, TriangleFeatures),
    "cup_and_handle": (CupAndHandleDetector, CupAndHandleFeatures),
}


class DatasetBuilder:
    def __init__(self, config_path: str | None = None):
        self.config = load_config(config_path) if config_path else load_config()

    def _load_detector_and_features(self, pattern: str):
        if pattern not in DETECTOR_REGISTRY:
            raise ValueError(f"Unknown pattern '{pattern}'")
        detector_cls, feature_cls = DETECTOR_REGISTRY[pattern]
        pattern_cfg = self.config["patterns"][pattern]
        detector = detector_cls(pattern_cfg.get("detector", {}))
        features = feature_cls()
        return detector, features

    def _prepare_dataframe(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, str]:
        if "symbol" in df.columns and "close" in df.columns:
            df = df[df["symbol"] == symbol].copy()
            if len(df) == 0:
                raise ValueError(f"No data found for symbol '{symbol}'")
            df = df.sort_values("timestamp").reset_index(drop=True)
            ts_col = "timestamp"
        else:
            ts_col = "TimeStamp" if "TimeStamp" in df.columns else "timestamp"
            if ts_col not in df.columns or symbol not in df.columns:
                raise ValueError(f"Expected wide format with column '{symbol}' and timestamp column")
            df = pd.DataFrame({"timestamp": df[ts_col], "close": df[symbol]})
            df = df.dropna(subset=["close"]).reset_index(drop=True)
            ts_col = "timestamp"
        return df, ts_col

    def build(self, pattern: str, symbol: str, input_path: Optional[str] = None) -> str:
        cfg = self.config
        pattern_cfg = cfg["patterns"][pattern]
        data_cfg = cfg["data"]
        trading_cfg = cfg.get("trading_hours", {})

        detector, feature_extractor = self._load_detector_and_features(pattern)

        source_path = input_path or data_cfg["input_path"]
        df = pd.read_csv(source_path)
        df, ts_col = self._prepare_dataframe(df, symbol)

        window_bars = pattern_cfg["window_bars"]
        is_daily = pattern_cfg.get("data_frequency", "intraday") == "daily"

        if not is_daily and trading_cfg.get("filter_enabled", True):
            df["ts"] = pd.to_datetime(df[ts_col])
            open_t = trading_cfg.get("market_open", "09:30")
            close_t = trading_cfg.get("market_close", "16:00")
            mask = df["ts"].apply(
                lambda t: is_within_trading_hours(
                    t,
                    t,
                    market_open=open_t,
                    market_close=close_t,
                )
            )
            df = df[mask].reset_index(drop=True)
        else:
            df["ts"] = pd.to_datetime(df[ts_col])

        if len(df) < window_bars:
            raise ValueError(f"Not enough data for windows of size {window_bars}")

        output_dir = data_cfg["output_dir"]
        filename = f"{symbol}_{pattern}_windows.csv"
        output_path = resolve_output_path(output_dir, filename)

        columns = ["symbol", "start_timestamp", "end_timestamp"] + feature_extractor.feature_names + [
            pattern_cfg["label_column"]
        ]
        write_header(output_path, columns)

        batch: List[dict] = []
        BATCH_SIZE = 500
        total_written = 0

        for end_idx in range(window_bars - 1, len(df)):
            start_idx = end_idx - window_bars + 1
            window = df.iloc[start_idx : end_idx + 1]
            close_window = window["close"]
            start_ts = window.iloc[0]["ts"]
            end_ts = window.iloc[-1]["ts"]

            if (not is_daily) and (start_ts.date() != end_ts.date()):
                continue

            feats = feature_extractor.compute_features(close_window)
            label = int(detector.detect(close_window, window_bars=window_bars))

            row = {
                "symbol": symbol,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                **feats,
                pattern_cfg["label_column"]: label,
            }
            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                total_written += append_rows(output_path, columns, batch)
                batch = []

        if batch:
            total_written += append_rows(output_path, columns, batch)

        return output_path

