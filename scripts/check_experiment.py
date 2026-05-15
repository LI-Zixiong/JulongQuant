"""
Sanity checks for experiment_001 outputs.

This script does not retrain models. It validates saved predictions,
backtest outputs, return alignment, and compares top_n with equal_all baseline.
Results are printed to terminal and appended to the experiment report.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_experiment import (
    ExperimentConfig,
    _load_experiment_raw_data,
    build_returns_frame_from_next_target,
)
from src.backtest.engine import run_backtest
from src.backtest.metrics import (
    cumulative_nav,
    prediction_ic_summary,
    summarize_backtest,
)
from src.backtest.portfolio import PortfolioConfig
from src.data.preprocess import PreprocessConfig, preprocess_panel_data


def _read_series_csv(path: Path, name: str) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError(f"{name} is empty: {path}")

    if df.shape[1] != 1:
        raise ValueError(f"{name} should have exactly one value column, got {df.shape[1]}")

    series = df.iloc[:, 0].astype(float)
    series.name = name

    return series


def _read_weights_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    weights = pd.read_csv(path, index_col=0, parse_dates=True)

    if weights.empty:
        raise ValueError(f"daily_weights is empty: {path}")

    return weights.astype(float)


def _check_prediction_file(pred_df: pd.DataFrame, config: ExperimentConfig) -> dict:
    required_cols = [
        config.date_col,
        config.stock_col,
        "y_true",
        "y_pred",
        "model_name",
    ]

    missing_cols = [col for col in required_cols if col not in pred_df.columns]
    if missing_cols:
        raise ValueError(f"predictions missing required columns: {missing_cols}")

    if pred_df.empty:
        raise ValueError("predictions are empty")

    pred_df[config.date_col] = pd.to_datetime(pred_df[config.date_col])

    if pred_df[[config.date_col, config.stock_col]].isna().any().any():
        raise ValueError("predictions contain missing date or stock_id")

    if pred_df.duplicated(subset=[config.date_col, config.stock_col]).any():
        raise ValueError("predictions contain duplicated date-stock rows")

    if not np.isfinite(pred_df["y_pred"].to_numpy(dtype=float)).all():
        raise ValueError("y_pred contains NaN or Inf values")

    if not np.isfinite(pred_df["y_true"].to_numpy(dtype=float)).all():
        raise ValueError("y_true contains NaN or Inf values")

    print("Prediction file check passed.")
    print(f"Prediction rows: {len(pred_df):,}")
    print(
        "Prediction date range: "
        f"{pred_df[config.date_col].min()} to {pred_df[config.date_col].max()}"
    )
    print("y_pred summary:")
    print(pred_df["y_pred"].describe())

    y_pred_stats = pred_df["y_pred"].describe()
    return {
        "n_rows": len(pred_df),
        "date_min": str(pred_df[config.date_col].min()),
        "date_max": str(pred_df[config.date_col].max()),
        "y_pred_mean": float(y_pred_stats["mean"]),
        "y_pred_std": float(y_pred_stats["std"]),
        "y_pred_min": float(y_pred_stats["min"]),
        "y_pred_max": float(y_pred_stats["max"]),
    }


def _check_backtest_outputs(
    daily_returns: pd.Series,
    daily_nav: pd.Series,
    daily_weights: pd.DataFrame,
    daily_turnover: pd.Series,
    config: ExperimentConfig,
) -> None:
    if daily_returns.empty:
        raise ValueError("daily_returns is empty")

    if daily_nav.empty:
        raise ValueError("daily_nav is empty")

    if daily_weights.empty:
        raise ValueError("daily_weights is empty")

    if daily_turnover.empty:
        raise ValueError("daily_turnover is empty")

    if not daily_returns.index.is_monotonic_increasing:
        raise ValueError("daily_returns index is not sorted")

    if not daily_nav.index.is_monotonic_increasing:
        raise ValueError("daily_nav index is not sorted")

    if not daily_weights.index.is_monotonic_increasing:
        raise ValueError("daily_weights index is not sorted")

    if not daily_turnover.index.is_monotonic_increasing:
        raise ValueError("daily_turnover index is not sorted")

    if not np.isfinite(daily_returns.to_numpy()).all():
        raise ValueError("daily_returns contains NaN or Inf values")

    if not np.isfinite(daily_nav.to_numpy()).all():
        raise ValueError("daily_nav contains NaN or Inf values")

    if not np.isfinite(daily_weights.fillna(0.0).to_numpy()).all():
        raise ValueError("daily_weights contains NaN or Inf values after fillna(0)")

    if not np.isfinite(daily_turnover.to_numpy()).all():
        raise ValueError("daily_turnover contains NaN or Inf values")

    expected_nav = cumulative_nav(daily_returns)

    if not np.allclose(
        daily_nav.to_numpy(),
        expected_nav.to_numpy(),
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError("daily_nav is inconsistent with cumulative_nav(daily_returns)")

    weights_filled = daily_weights.fillna(0.0)
    weight_sums = weights_filled.sum(axis=1)

    if not np.allclose(weight_sums.to_numpy(), 1.0, rtol=1e-8, atol=1e-10):
        bad_dates = weight_sums[~np.isclose(weight_sums, 1.0)].head()
        raise ValueError(f"Some daily weights do not sum to 1.0: {bad_dates}")

    if (weights_filled < -1e-12).any().any():
        raise ValueError("daily_weights contains negative weights")

    nonzero_counts = (weights_filled > 0.0).sum(axis=1)

    if (nonzero_counts > config.top_n).any():
        bad_counts = nonzero_counts[nonzero_counts > config.top_n].head()
        raise ValueError(f"Some top_n portfolios hold more than top_n stocks: {bad_counts}")

    if (daily_turnover < -1e-12).any():
        raise ValueError("daily_turnover contains negative values")

    if len(daily_returns) != len(daily_weights):
        raise ValueError(
            "daily_returns and daily_weights length mismatch: "
            f"{len(daily_returns)} vs {len(daily_weights)}"
        )

    if len(daily_returns) != len(daily_turnover):
        raise ValueError(
            "daily_returns and daily_turnover length mismatch: "
            f"{len(daily_returns)} vs {len(daily_turnover)}"
        )

    if daily_returns.index.min() <= daily_weights.index.min():
        print(
            "Return/weight date check: first return date is after or equal to first signal date."
        )

    print("Backtest output check passed.")
    print(f"Backtest periods: {len(daily_returns):,}")
    print(f"Daily return date range: {daily_returns.index.min()} to {daily_returns.index.max()}")
    print(f"Weight signal date range: {daily_weights.index.min()} to {daily_weights.index.max()}")
    print(f"Mean turnover: {daily_turnover.mean():.6f}")

    return {
        "n_periods": len(daily_returns),
        "return_date_min": str(daily_returns.index.min()),
        "return_date_max": str(daily_returns.index.max()),
        "weight_date_min": str(daily_weights.index.min()),
        "weight_date_max": str(daily_weights.index.max()),
        "mean_turnover": float(daily_turnover.mean()),
    }


def _rebuild_clean_df_and_returns(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = _load_experiment_raw_data(config.data_path)

    if config.date_col not in raw_df.columns and config.date_col in raw_df.index.names:
        raw_df = raw_df.reset_index()

    preprocess_config = PreprocessConfig(
        date_col=config.date_col,
        stock_col=config.stock_col,
        feature_cols=list(config.feature_cols),
        target_col=config.target_col,
        meta_cols=list(config.meta_cols),
        replace_inf_with_nan=True,
        drop_rows_with_missing_keys=True,
        drop_rows_with_missing_features=False,
        drop_rows_with_missing_target=True,
        duplicate_policy="raise",
        sort_values=True,
    )

    preprocess_result = preprocess_panel_data(
        df=raw_df,
        config=preprocess_config,
    )

    clean_df = preprocess_result.df

    returns_df = build_returns_frame_from_next_target(
        df=clean_df,
        date_col=config.date_col,
        stock_col=config.stock_col,
        target_col=config.target_col,
        return_col=config.return_col,
    )

    returns_df[config.date_col] = pd.to_datetime(returns_df[config.date_col])
    returns_df[config.stock_col] = returns_df[config.stock_col].astype(str)

    return clean_df, returns_df


def _check_return_alignment(
    pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    config: ExperimentConfig,
    n_samples: int = 5,
) -> list[dict]:
    pred_work = pred_df.copy()
    pred_work[config.date_col] = pd.to_datetime(pred_work[config.date_col])
    pred_work[config.stock_col] = pred_work[config.stock_col].astype(str)

    return_work = returns_df.copy()
    return_work[config.date_col] = pd.to_datetime(return_work[config.date_col])
    return_work[config.stock_col] = return_work[config.stock_col].astype(str)

    pred_dates = pd.DatetimeIndex(pred_work[config.date_col].unique()).sort_values()
    return_dates = pd.DatetimeIndex(return_work[config.date_col].unique()).sort_values()

    if pred_dates.empty:
        raise ValueError("No prediction dates found")

    if return_dates.empty:
        raise ValueError("No return dates found")

    print("Return alignment samples:")
    samples: list[dict] = []

    for signal_date in pred_dates[:n_samples]:
        position = return_dates.searchsorted(signal_date, side="right")

        if position >= len(return_dates):
            continue

        return_date = pd.Timestamp(return_dates[position])

        daily_pred = pred_work[pred_work[config.date_col] == signal_date]
        daily_returns = return_work[return_work[config.date_col] == return_date]

        pred_stocks = set(daily_pred[config.stock_col])
        return_stocks = set(daily_returns[config.stock_col])
        overlap = pred_stocks & return_stocks

        print(
            f"signal_date={signal_date}, return_date={return_date}, "
            f"pred_stocks={len(pred_stocks)}, return_stocks={len(return_stocks)}, "
            f"overlap={len(overlap)}"
        )

        samples.append({
            "signal_date": str(signal_date),
            "return_date": str(return_date),
            "pred_stocks": len(pred_stocks),
            "return_stocks": len(return_stocks),
            "overlap": len(overlap),
        })

        if not overlap:
            raise ValueError(
                f"No stock overlap between signal_date={signal_date} "
                f"and return_date={return_date}"
            )

    print("Return alignment check passed.")
    return samples


def _compare_with_equal_all(
    pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    top_n_summary: dict,
    config: ExperimentConfig,
) -> dict:
    equal_config = PortfolioConfig(
        strategy="equal_all",
        top_n=config.top_n,
        pred_col="y_pred",
        stock_col=config.stock_col,
    )

    equal_result = run_backtest(
        pred_df=pred_df,
        returns_df=returns_df,
        portfolio_config=equal_config,
        return_col=config.return_col,
        date_col=config.date_col,
        stock_col=config.stock_col,
        periods_per_year=config.periods_per_year,
    )

    equal_summary = equal_result["summary"]

    print("\nTop-N vs Equal-All baseline:")
    print(
        f"Top-N Sharpe:     {top_n_summary['sharpe_ratio']:.6f} | "
        f"Equal-All Sharpe: {equal_summary['sharpe_ratio']:.6f}"
    )
    print(
        f"Top-N Final NAV:  {top_n_summary['final_nav']:.6f} | "
        f"Equal-All NAV:    {equal_summary['final_nav']:.6f}"
    )
    print(
        f"Top-N AnnRet:     {top_n_summary['annualized_return']:.6f} | "
        f"Equal-All AnnRet: {equal_summary['annualized_return']:.6f}"
    )
    print(
        f"Top-N MaxDD:      {top_n_summary['max_drawdown']:.6f} | "
        f"Equal-All MaxDD:  {equal_summary['max_drawdown']:.6f}"
    )
    print(
        f"Top-N Turnover:   {top_n_summary['mean_turnover']:.6f} | "
        f"Equal Turnover:   {equal_summary['mean_turnover']:.6f}"
    )

    return {
        "top_n_sharpe": top_n_summary["sharpe_ratio"],
        "equal_sharpe": equal_summary["sharpe_ratio"],
        "top_n_nav": top_n_summary["final_nav"],
        "equal_nav": equal_summary["final_nav"],
        "top_n_annret": top_n_summary["annualized_return"],
        "equal_annret": equal_summary["annualized_return"],
        "top_n_maxdd": top_n_summary["max_drawdown"],
        "equal_maxdd": equal_summary["max_drawdown"],
        "top_n_turnover": top_n_summary["mean_turnover"],
        "equal_turnover": equal_summary["mean_turnover"],
    }


def _append_sanity_report(
    prediction_check: dict,
    backtest_check: dict,
    alignment_samples: list[dict],
    comparison: dict,
    config: ExperimentConfig,
    ic_summary: dict | None = None,
) -> Path:
    report_path = Path(config.report_path)
    existing = report_path.read_text(encoding="utf-8")

    md = "\n\n## Sanity Checks\n\n"

    md += "### Prediction File\n\n"
    md += "| Check | Value |\n"
    md += "|---|---|\n"
    md += f"| rows | {prediction_check['n_rows']:,} |\n"
    md += f"| date range | {prediction_check['date_min']} to {prediction_check['date_max']} |\n"
    md += f"| y_pred mean | {prediction_check['y_pred_mean']:.6f} |\n"
    md += f"| y_pred std | {prediction_check['y_pred_std']:.6f} |\n"
    md += f"| y_pred min | {prediction_check['y_pred_min']:.6f} |\n"
    md += f"| y_pred max | {prediction_check['y_pred_max']:.6f} |\n"

    md += "\n### Backtest Outputs\n\n"
    md += "| Check | Value |\n"
    md += "|---|---|\n"
    md += f"| backtest periods | {backtest_check['n_periods']:,} |\n"
    md += f"| return date range | {backtest_check['return_date_min']} to {backtest_check['return_date_max']} |\n"
    md += f"| weight date range | {backtest_check['weight_date_min']} to {backtest_check['weight_date_max']} |\n"
    md += f"| mean turnover | {backtest_check['mean_turnover']:.6f} |\n"

    md += "\n### Return Alignment Samples\n\n"
    md += "| signal_date | return_date | pred_stocks | return_stocks | overlap |\n"
    md += "|---|---|---|---|---|\n"
    for row in alignment_samples:
        md += (
            f"| {row['signal_date']} | {row['return_date']} | "
            f"{row['pred_stocks']:,} | {row['return_stocks']:,} | "
            f"{row['overlap']:,} |\n"
        )

    md += "\n### Top-N vs Equal-All Baseline\n\n"
    md += "| Metric | Top-N | Equal-All |\n"
    md += "|---|---|---|\n"
    md += f"| sharpe_ratio | {comparison['top_n_sharpe']:.6f} | {comparison['equal_sharpe']:.6f} |\n"
    md += f"| final_nav | {comparison['top_n_nav']:.6f} | {comparison['equal_nav']:.6f} |\n"
    md += f"| annualized_return | {comparison['top_n_annret']:.6f} | {comparison['equal_annret']:.6f} |\n"
    md += f"| max_drawdown | {comparison['top_n_maxdd']:.6f} | {comparison['equal_maxdd']:.6f} |\n"
    md += f"| mean_turnover | {comparison['top_n_turnover']:.6f} | {comparison['equal_turnover']:.6f} |\n"

    if ic_summary is not None:
        md += "\n### IC / Rank IC Summary\n\n"
        md += "| Metric | Value |\n"
        md += "|---|---|\n"
        ic_keys = [
            ("ic_n_periods", "IC periods"),
            ("ic_mean", "IC mean"),
            ("ic_std", "IC std"),
            ("ic_ir", "IC IR"),
            ("ic_positive_rate", "IC positive rate"),
            ("rank_ic_n_periods", "Rank IC periods"),
            ("rank_ic_mean", "Rank IC mean"),
            ("rank_ic_std", "Rank IC std"),
            ("rank_ic_ir", "Rank IC IR"),
            ("rank_ic_positive_rate", "Rank IC positive rate"),
        ]
        for key, label in ic_keys:
            value = ic_summary[key]
            if isinstance(value, float):
                md += f"| {label} | {value:.6f} |\n"
            else:
                md += f"| {label} | {value} |\n"

    md += "\n### Interpretation\n\n"
    sharpe_delta = comparison["top_n_sharpe"] - comparison["equal_sharpe"]
    if sharpe_delta > 0:
        md += (
            f"- Top-N Sharpe is {sharpe_delta:.2f} higher than equal-all, "
            "indicating the model has predictive signal.\n"
        )
    else:
        md += (
            f"- Top-N Sharpe is {sharpe_delta:.2f} lower than equal-all. "
            "Model signal may be weak or negative at this horizon.\n"
        )
    if ic_summary is not None:
        ic_mean = ic_summary["ic_mean"]
        if ic_mean > 0:
            md += (
                f"- Mean IC is positive ({ic_mean:.4f}), confirming "
                "predictions are directionally aligned with realized returns.\n"
            )
        else:
            md += (
                f"- Mean IC is {ic_mean:.4f} (not meaningfully positive), "
                "suggesting weak cross-sectional predictive power.\n"
            )
    md += (
        f"- Top-N portfolios hold at most {config.top_n} stocks, "
        f"compared to the full universe in equal-all, "
        "resulting in higher concentration risk (wider drawdowns).\n"
    )
    md += (
        f"- Mean turnover of {comparison['top_n_turnover']:.2%} per period "
        "reflects daily rebalancing with a fixed top-N strategy. "
        "Transaction costs are not modeled.\n"
    )

    md += "\n### Known Limitations\n\n"
    md += "- V0 preprocessing: no factor standardization, winsorization, or industry neutralization.\n"
    md += "- Target `1d_next_raw` is raw next-day return including market beta and style factor exposure.\n"
    md += "- Backtest excludes transaction costs, slippage, and short-sale constraints.\n"
    md += "- 70/10/20 date split is fixed; no walk-forward or purged cross-validation.\n"
    md += "- Predictions use a single LightGBM model; no ensemble or meta-model weighting.\n"

    report_path.write_text(existing.rstrip() + md, encoding="utf-8")
    return report_path


def main() -> None:
    config = ExperimentConfig()

    output_dir = Path(config.output_dir)

    prediction_path = output_dir / "predictions_lightgbm.parquet"
    daily_returns_path = output_dir / "daily_returns_lightgbm.csv"
    daily_nav_path = output_dir / "daily_nav_lightgbm.csv"
    daily_weights_path = output_dir / "daily_weights_lightgbm.csv"
    daily_turnover_path = output_dir / "daily_turnover_lightgbm.csv"

    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {prediction_path}")

    pred_df = pd.read_parquet(prediction_path)
    pred_df[config.date_col] = pd.to_datetime(pred_df[config.date_col])
    pred_df[config.stock_col] = pred_df[config.stock_col].astype(str)

    daily_returns = _read_series_csv(daily_returns_path, "strategy_return")
    daily_nav = _read_series_csv(daily_nav_path, "nav")
    daily_weights = _read_weights_csv(daily_weights_path)
    daily_turnover = _read_series_csv(daily_turnover_path, "turnover")

    prediction_check = _check_prediction_file(pred_df, config)

    # Compute and print prediction IC summary (require at least 50 obs per date)
    pred_ic_summary: dict | None = None
    try:
        pred_ic_summary = prediction_ic_summary(
            pred_df=pred_df,
            date_col=config.date_col,
            y_true_col="y_true",
            y_pred_col="y_pred",
            min_obs=50,
        )

        print("\nPrediction IC Summary:")
        for k, v in pred_ic_summary.items():
            print(f"{k:30s} | {v}")
    except Exception as exc:  # pragma: no cover - best-effort summary
        print(f"Prediction IC summary could not be computed: {exc}")

    backtest_check = _check_backtest_outputs(
        daily_returns=daily_returns,
        daily_nav=daily_nav,
        daily_weights=daily_weights,
        daily_turnover=daily_turnover,
        config=config,
    )

    _, returns_df = _rebuild_clean_df_and_returns(config)

    alignment_samples = _check_return_alignment(
        pred_df=pred_df,
        returns_df=returns_df,
        config=config,
    )

    top_n_summary = summarize_backtest(
        returns=daily_returns,
        periods_per_year=config.periods_per_year,
    )

    top_n_summary["mean_turnover"] = float(daily_turnover.mean())
    top_n_summary["total_turnover"] = float(daily_turnover.sum())

    comparison = _compare_with_equal_all(
        pred_df=pred_df,
        returns_df=returns_df,
        top_n_summary=top_n_summary,
        config=config,
    )

    report_path = _append_sanity_report(
        prediction_check=prediction_check,
        backtest_check=backtest_check,
        alignment_samples=alignment_samples,
        comparison=comparison,
        config=config,
        ic_summary=pred_ic_summary,
    )

    print(f"\nSanity checks appended to: {report_path}")
    print("All experiment sanity checks passed.")


if __name__ == "__main__":
    main()