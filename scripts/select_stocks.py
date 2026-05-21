"""
Apply select_stocks logic to factor_panel_12.parquet.
Output factor_panel_1500.parquet for experiments.

Selection pipeline:
  1. Hard filters: ST, price, valid_days, valid_ratio, mega-cap
  2. Composite scoring: data_quality(0.20) + liquidity(0.40) + stability(0.20) + size(0.20)
  3. Industry-stratified top-1500 selection
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INPUT = "dataset/processed/factor_panel_12.parquet"
OUTPUT = "dataset/processed/factor_panel_1500.parquet"
TARGET = 1500
MAX_MKTCAP_Q = 0.95


def _pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    pct = s.rank(pct=True)
    if not higher_is_better:
        pct = 1.0 - pct
    return pct.fillna(0.0)


def main():
    df = pd.read_parquet(INPUT)
    df["time"] = pd.to_datetime(df["time"]).dt.normalize()
    df["stock_id"] = df["stock_id"].astype(str).str.strip()
    print(f"Loaded: {len(df):,} rows, {df['stock_id'].nunique()} stocks")

    # ---- Per-stock metrics ----
    grouped = df.groupby("stock_id")
    n_valid = grouped["SIZE"].count()
    n_dates = grouped["time"].nunique()
    valid_ratio = n_valid / n_dates
    mktcap = grouped["mktcap_total"].median()
    liq = grouped["LIQUIDITY"].median()
    resvol = grouped["RESVOL"].median()
    industry = grouped["industry_csrc_2012"].last()

    metrics = pd.DataFrame({
        "stock_id": n_valid.index,
        "n_valid_days": n_valid.values,
        "valid_ratio": valid_ratio.values,
        "mktcap_median": mktcap.values,
        "liq_median": liq.values,
        "resvol_median": resvol.values,
        "industry": industry.values,
    })
    n_total = len(metrics)
    print(f"Stocks: {n_total}")

    # ---- Hard filters ----
    keep = np.ones(len(metrics), dtype=bool)
    stats = {}

    # Mega-cap: top 5%
    cutoff = metrics["mktcap_median"].quantile(MAX_MKTCAP_Q)
    mask = metrics["mktcap_median"] > cutoff
    stats["mega_cap"] = int(mask.sum())
    keep[mask.values] = False

    # valid_ratio < 75%
    mask = metrics["valid_ratio"] < 0.75
    stats["valid_ratio"] = int(mask.sum())
    keep[mask.values] = False

    # n_valid_days < 180
    mask = metrics["n_valid_days"] < 180
    stats["n_valid_days"] = int(mask.sum())
    keep[mask.values] = False

    metrics = metrics.loc[keep].copy().reset_index(drop=True)
    print(f"After hard filters: {len(metrics)} stocks")
    for k, v in stats.items():
        print(f"  {k}: removed {v}")

    # ---- Composite score ----
    metrics["score"] = (
        _pct_rank(metrics["valid_ratio"]) * 0.10
        + _pct_rank(metrics["n_valid_days"]) * 0.10
        + _pct_rank(metrics["liq_median"]) * 0.40
        + _pct_rank(metrics["resvol_median"], higher_is_better=False) * 0.20
        + _pct_rank(metrics["mktcap_median"]) * 0.20
    )

    # ---- Industry-stratified selection ----
    ind_counts = metrics["industry"].value_counts()
    ind_weights = ind_counts / ind_counts.sum()
    allocation = (ind_weights * TARGET).round().astype(int)
    diff = TARGET - allocation.sum()
    for i in range(abs(diff)):
        allocation.iloc[allocation.argmax()] += np.sign(diff)

    selected = []
    for ind, quota in allocation.items():
        if quota <= 0:
            continue
        pool = metrics[metrics["industry"] == ind]
        top = pool.nlargest(quota, "score")
        selected.extend(top["stock_id"].tolist())

    print(f"Selected: {len(selected)} stocks")

    # ---- Filter factor panel ----
    out = df[df["stock_id"].isin(selected)].copy()
    print(f"Output: {len(out):,} rows, {out['stock_id'].nunique()} stocks")

    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT, index=False)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
