"""
Simple equal-weight ensemble — full test period, same backtest as check_experiment.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.engine import run_backtest
from src.backtest.portfolio import PortfolioConfig
from scripts.run_experiment import (
    ExperimentConfig, _load_experiment_raw_data,
    build_returns_frame_from_next_target,
)
from src.data.preprocess import PreprocessConfig, preprocess_panel_data

OUTPUT_DIR = Path("dataset/output/experiment_001")
MODELS = ["lightgbm", "xgboost", "dlinear", "itransformer", "tsmixer"]


def main():
    # Load all predictions
    preds = {}
    for m in MODELS:
        p = OUTPUT_DIR / f"predictions_{m}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["time"] = pd.to_datetime(df["time"]).dt.normalize()
            df["stock_id"] = df["stock_id"].astype(str).str.strip()
            preds[m] = df
            print(f"  {m:15s}: {len(df):,} rows, {df['time'].nunique()} dates")

    available = sorted(preds.keys())

    # Align all models on common (time, stock_id)
    merged = preds[available[0]][["time", "stock_id", "y_true"]].copy()
    for m in available:
        df = preds[m][["time", "stock_id", "y_pred"]].rename(columns={"y_pred": m})
        merged = merged.merge(df, on=["time", "stock_id"], how="inner")
    print(f"\nMerged: {len(merged):,} rows, {merged['time'].nunique()} dates")

    # Build returns frame
    config = ExperimentConfig()
    raw_df = _load_experiment_raw_data(config.data_path)
    if config.date_col not in raw_df.columns and config.date_col in raw_df.index.names:
        raw_df = raw_df.reset_index()
    clean_df = preprocess_panel_data(raw_df, PreprocessConfig(
        date_col=config.date_col, stock_col=config.stock_col,
        feature_cols=list(config.feature_cols), target_col=config.target_col,
        meta_cols=list(config.meta_cols),
        replace_inf_with_nan=True, drop_rows_with_missing_keys=True,
        drop_rows_with_missing_target=True, duplicate_policy="raise", sort_values=True,
    )).df
    returns = build_returns_frame_from_next_target(
        clean_df, config.date_col, config.stock_col,
        config.backtest_return_source, config.return_col,
    )
    returns["time"] = pd.to_datetime(returns["time"]).dt.normalize()
    returns["stock_id"] = returns["stock_id"].astype(str).str.strip()

    from sklearn.linear_model import RidgeCV
    X = merged[available].values
    y = merged["y_true"].values
    ridge = RidgeCV(alphas=[0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0], fit_intercept=False)
    ridge.fit(X, y)
    w = pd.Series(ridge.coef_, index=available)
    print(f"\nRidgeCV alpha={ridge.alpha_:.1f}  weights:")
    for m, wv in w.items():
        print(f"  {m:15s}: {wv:+.4f}")

    # Equal-weight ensemble
    merged["y_pred_eq"] = merged[available].mean(axis=1)
    merged["y_pred_ridge"] = X @ ridge.coef_

    pfolio = PortfolioConfig(strategy="top_n", top_n=50, pred_col="y_pred", stock_col="stock_id")

    for label, col in [("RidgeCV", "y_pred_ridge"), ("Equal-Weight", "y_pred_eq")]:
        df = merged[["time", "stock_id", "y_true"]].copy()
        df["y_pred"] = merged[col]
        result = run_backtest(
            pred_df=df, returns_df=returns, portfolio_config=pfolio,
            return_col="return_1d", date_col="time", stock_col="stock_id",
        )
        s = result["summary"]
        print(f"\n=== {label} ===")
        print(f"  Sharpe={s['sharpe_ratio']:.4f}  AnnRet={s['annualized_return']:.4f}  "
              f"MaxDD={s['max_drawdown']:.4f}  NAV={s['final_nav']:.4f}  "
              f"Turnover={s['mean_turnover']:.4f}  WinRate={s['hit_rate']:.4f}")

    # Per-model on same (date, stock) universe for fair comparison
    print(f"\n=== Individual Models (same universe) ===")
    for m in available:
        df = merged[["time", "stock_id"]].merge(
            preds[m][["time", "stock_id", "y_pred"]], on=["time", "stock_id"]
        )
        r = run_backtest(
            pred_df=df, returns_df=returns, portfolio_config=pfolio,
            return_col="return_1d", date_col="time", stock_col="stock_id",
        )
        s = r["summary"]
        print(f"  {m:15s}: Sharpe={s['sharpe_ratio']:.4f}  AnnRet={s['annualized_return']:.4f}  "
              f"MaxDD={s['max_drawdown']:.4f}  NAV={s['final_nav']:.4f}  "
              f"Turnover={s['mean_turnover']:.4f}  WinRate={s['hit_rate']:.4f}")

    # Save
    out = merged[["time", "stock_id", "y_true"]].copy()
    out["y_pred"] = merged["y_pred_ridge"]
    out["model_name"] = "ensemble"
    out.to_parquet(OUTPUT_DIR / "predictions_ensemble.parquet", index=False)
    print(f"\nSaved: {OUTPUT_DIR / 'predictions_ensemble.parquet'}")


if __name__ == "__main__":
    main()
