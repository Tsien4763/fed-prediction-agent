"""FOMC label loading interface. Historical records intentionally excluded.

This code-only export is modified from the MIT public snapshot. Supply your
own authorized CSV via an explicit path or FOMC_LABELS_CSV. No data bundled.
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

FOMC_LABELS: list[dict] = []  # Historical records excluded from this export.

def build_fomc_label_df(path: str | Path | None = None) -> pd.DataFrame:
    source = path or os.environ.get("FOMC_LABELS_CSV")
    if source is None:
        raise FileNotFoundError("No labels bundled. Supply an authorized CSV via FOMC_LABELS_CSV or path.")
    df = pd.read_csv(source)
    required = {"date", "decision", "ff_lower", "ff_upper", "desc"}
    if not required.issubset(df.columns):
        raise ValueError("FOMC label CSV must contain date, decision, ff_lower, ff_upper, desc")
    if df.empty:
        raise ValueError("FOMC label CSV is empty")
    df["date"] = pd.to_datetime(df["date"])
    df["quarter"] = df["date"].dt.to_period("Q").dt.to_timestamp(how="end")
    return df
