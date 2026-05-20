"""
Sanity checks for multi-model experiment_001 outputs.

This script does not retrain models. It validates saved predictions,
backtest outputs, return alignment, IC / Rank IC, and equal-all baseline
comparison for each configured model.
"""

from pathlib import Path
from typing import Any

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
        raise ValueError(
            f"{name} should have exactly one value column, got {df.shape[1]}"
        )

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


def _load_model_outputs(
    output_dir: Path,
    model_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    prediction_path = output_dir / f"predictions_{model_name}.parquet"
    daily_returns_path = output_dir / f"daily_returns_{model_name}.csv"
    daily_nav_path = output_dir / f"daily_nav_{model_name}.csv"
    daily_weights_path = output_dir / f"daily_weights_{model_name}.csv"
    daily_turnover_path = output_dir / f"daily_turnover_{model_name}.csv"

    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {prediction_path}")

    pred_df = pd.read_parquet(prediction_path)
    daily_returns = _read_series_csv(daily_returns_path, "strategy_return")
    daily_nav = _read_series_csv(daily_nav_path, "nav")
    daily_weights = _read_weights_csv(daily_weights_path)
    daily_turnover = _read_series_csv(daily_turnover_path, "turnover")

    return pred_df, daily_returns, daily_nav, daily_weights, daily_turnover


def _check_prediction_file(
    pred_df: pd.DataFrame,
    config: ExperimentConfig,
    model_name: str,
) -> None:
    required_cols = [
        config.date_col,
        config.stock_col,
        "y_true",
        "y_pred",
        "model_name",
    ]

    missing_cols = [col for col in required_cols if col not in pred_df.columns]
    if missing_cols:
        raise ValueError(
            f"{model_name} predictions missing required columns: {missing_cols}"
        )

    if pred_df.empty:
        raise ValueError(f"{model_name} predictions are empty")

    pred_df[config.date_col] = pd.to_datetime(
        pred_df[config.date_col],
        errors="raise",
    )
    pred_df[config.stock_col] = pred_df[config.stock_col].astype(str)

    if pred_df[[config.date_col, config.stock_col]].isna().any().any():
        raise ValueError(f"{model_name} predictions contain missing date or stock_id")

    if pred_df.duplicated(subset=[config.date_col, config.stock_col]).any():
        raise ValueError(f"{model_name} predictions contain duplicated date-stock rows")

    if not np.isfinite(pred_df["y_pred"].to_numpy(dtype=float)).all():
        raise ValueError(f"{model_name} y_pred contains NaN or Inf values")

    if not np.isfinite(pred_df["y_true"].to_numpy(dtype=float)).all():
        raise ValueError(f"{model_name} y_true contains NaN or Inf values")


def _check_backtest_outputs(
    daily_returns: pd.Series,
    daily_nav: pd.Series,
    daily_weights: pd.DataFrame,
    daily_turnover: pd.Series,
    config: ExperimentConfig,
    model_name: str,
) -> None:
    if daily_returns.empty:
        raise ValueError(f"{model_name} daily_returns is empty")

    if daily_nav.empty:
        raise ValueError(f"{model_name} daily_nav is empty")

    if daily_weights.empty:
        raise ValueError(f"{model_name} daily_weights is empty")

    if daily_turnover.empty:
        raise ValueError(f"{model_name} daily_turnover is empty")

    if not daily_returns.index.is_monotonic_increasing:
        raise ValueError(f"{model_name} daily_returns index is not sorted")

    if not daily_nav.index.is_monotonic_increasing:
        raise ValueError(f"{model_name} daily_nav index is not sorted")

    if not daily_weights.index.is_monotonic_increasing:
        raise ValueError(f"{model_name} daily_weights index is not sorted")

    if not daily_turnover.index.is_monotonic_increasing:
        raise ValueError(f"{model_name} daily_turnover index is not sorted")

    if not np.isfinite(daily_returns.to_numpy()).all():
        raise ValueError(f"{model_name} daily_returns contains NaN or Inf values")

    if not np.isfinite(daily_nav.to_numpy()).all():
        raise ValueError(f"{model_name} daily_nav contains NaN or Inf values")

    weights_filled = daily_weights.fillna(0.0)

    if not np.isfinite(weights_filled.to_numpy()).all():
        raise ValueError(
            f"{model_name} daily_weights contains NaN or Inf values after fillna(0)"
        )

    if not np.isfinite(daily_turnover.to_numpy()).all():
        raise ValueError(f"{model_name} daily_turnover contains NaN or Inf values")

    expected_nav = cumulative_nav(daily_returns)

    if not np.allclose(
        daily_nav.to_numpy(),
        expected_nav.to_numpy(),
        rtol=1e-8,
        atol=1e-10,
    ):
        raise ValueError(
            f"{model_name} daily_nav is inconsistent with cumulative_nav(daily_returns)"
        )

    weight_sums = weights_filled.sum(axis=1)

    if not np.allclose(weight_sums.to_numpy(), 1.0, rtol=1e-8, atol=1e-10):
        bad_dates = weight_sums[~np.isclose(weight_sums, 1.0)].head()
        raise ValueError(
            f"{model_name} daily weights do not sum to 1.0: {bad_dates}"
        )

    if (weights_filled < -1e-12).any().any():
        raise ValueError(f"{model_name} daily_weights contains negative weights")

    nonzero_counts = (weights_filled > 0.0).sum(axis=1)

    if (nonzero_counts > config.top_n).any():
        bad_counts = nonzero_counts[nonzero_counts > config.top_n].head()
        raise ValueError(
            f"{model_name} top_n portfolios hold more than top_n stocks: "
            f"{bad_counts}"
        )

    if (daily_turnover < -1e-12).any():
        raise ValueError(f"{model_name} daily_turnover contains negative values")

    if len(daily_returns) != len(daily_weights):
        raise ValueError(
            f"{model_name} daily_returns and daily_weights length mismatch: "
            f"{len(daily_returns)} vs {len(daily_weights)}"
        )

    if len(daily_returns) != len(daily_turnover):
        raise ValueError(
            f"{model_name} daily_returns and daily_turnover length mismatch: "
            f"{len(daily_returns)} vs {len(daily_turnover)}"
        )


def _rebuild_clean_df_and_returns(
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
        target_col=config.backtest_return_source,
        return_col=config.return_col,
    )

    returns_df[config.date_col] = pd.to_datetime(
        returns_df[config.date_col],
        errors="raise",
    )
    returns_df[config.stock_col] = returns_df[config.stock_col].astype(str)

    return clean_df, returns_df


def _check_return_alignment(
    pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    config: ExperimentConfig,
    model_name: str,
    n_samples: int = 5,
) -> list[dict[str, Any]]:
    pred_work = pred_df.copy()
    pred_work[config.date_col] = pd.to_datetime(
        pred_work[config.date_col],
        errors="raise",
    )
    pred_work[config.stock_col] = pred_work[config.stock_col].astype(str)

    return_work = returns_df.copy()
    return_work[config.date_col] = pd.to_datetime(
        return_work[config.date_col],
        errors="raise",
    )
    return_work[config.stock_col] = return_work[config.stock_col].astype(str)

    pred_dates = pd.DatetimeIndex(pred_work[config.date_col].unique()).sort_values()
    return_dates = pd.DatetimeIndex(return_work[config.date_col].unique()).sort_values()

    if pred_dates.empty:
        raise ValueError(f"{model_name}: no prediction dates found")

    if return_dates.empty:
        raise ValueError(f"{model_name}: no return dates found")

    samples: list[dict[str, Any]] = []

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

        sample = {
            "model": model_name,
            "signal_date": pd.Timestamp(signal_date),
            "return_date": return_date,
            "pred_stocks": len(pred_stocks),
            "return_stocks": len(return_stocks),
            "overlap": len(overlap),
        }
        samples.append(sample)

        if not overlap:
            raise ValueError(
                f"{model_name}: no stock overlap between signal_date={signal_date} "
                f"and return_date={return_date}"
            )

    return samples


def _run_equal_all_baseline(
    pred_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    config: ExperimentConfig,
) -> dict[str, Any]:
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

    return equal_result["summary"]


def _summarize_saved_backtest(
    daily_returns: pd.Series,
    daily_turnover: pd.Series,
    config: ExperimentConfig,
) -> dict[str, Any]:
    summary = summarize_backtest(
        returns=daily_returns,
        periods_per_year=config.periods_per_year,
    )

    summary["mean_turnover"] = float(daily_turnover.mean())
    summary["total_turnover"] = float(daily_turnover.sum())

    return summary


def _check_model_comparison_file(
    comparison_df: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    required_cols = [
        "model",
        "train_rmse",
        "valid_rmse",
        "sharpe_ratio",
        "annualized_return",
        "max_drawdown",
        "final_nav",
        "mean_turnover",
    ]
    missing_cols = [col for col in required_cols if col not in comparison_df.columns]

    if missing_cols:
        raise ValueError(f"model_comparison.csv missing columns: {missing_cols}")

    expected_models = set(config.model_names)
    actual_models = set(comparison_df["model"].astype(str))

    if expected_models != actual_models:
        raise ValueError(
            f"model comparison model mismatch: expected {expected_models}, "
            f"got {actual_models}"
        )

    numeric_cols = [col for col in required_cols if col != "model"]
    values = comparison_df[numeric_cols].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("model_comparison.csv contains NaN or Inf values")


def _assert_summary_matches_comparison(
    model_name: str,
    summary: dict[str, Any],
    comparison_df: pd.DataFrame,
) -> None:
    row = comparison_df.loc[comparison_df["model"] == model_name]

    if len(row) != 1:
        raise ValueError(f"Expected exactly one comparison row for {model_name}")

    row = row.iloc[0]

    keys = [
        "sharpe_ratio",
        "annualized_return",
        "max_drawdown",
        "final_nav",
        "mean_turnover",
    ]

    for key in keys:
        if key not in row:
            continue

        if not np.isclose(float(summary[key]), float(row[key]), rtol=1e-8, atol=1e-10):
            raise ValueError(
                f"{model_name} summary mismatch for {key}: "
                f"summary={summary[key]}, comparison={row[key]}"
            )


def _write_sanity_report(
    report_path: Path,
    model_check_rows: list[dict[str, Any]],
    ic_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    alignment_rows: list[dict[str, Any]],
    comparison_df: pd.DataFrame,
) -> Path:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Experiment 001 Sanity Checks",
        "",
        "## Model Comparison",
        "",
        comparison_df.to_markdown(index=False),
        "",
        "## Model Output Checks",
        "",
        pd.DataFrame(model_check_rows).to_markdown(index=False),
        "",
        "## IC / Rank IC Summary",
        "",
        pd.DataFrame(ic_rows).to_markdown(index=False),
        "",
        "## Top-N vs Equal-All Baseline",
        "",
        pd.DataFrame(baseline_rows).to_markdown(index=False),
        "",
        "## Return Alignment Samples",
        "",
        pd.DataFrame(alignment_rows).to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        (
            "- Each model passed prediction, backtest output, weight, turnover, "
            "and return-alignment sanity checks."
        ),
        (
            "- Top-N results should be compared against the equal-all baseline "
            "to evaluate incremental model selection value."
        ),
        (
            "- IC and Rank IC measure prediction quality directly, while "
            "Sharpe and final NAV measure portfolio-level performance."
        ),
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main() -> None:
    config = ExperimentConfig()
    output_dir = Path(config.output_dir)

    comparison_path = output_dir / "model_comparison.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(f"Missing model comparison file: {comparison_path}")

    comparison_df = pd.read_csv(comparison_path)
    _check_model_comparison_file(comparison_df, config)

    _, returns_df = _rebuild_clean_df_and_returns(config)

    model_check_rows: list[dict[str, Any]] = []
    ic_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []

    for model_name in config.model_names:
        print(f"\nChecking model: {model_name}")

        pred_df, daily_returns, daily_nav, daily_weights, daily_turnover = (
            _load_model_outputs(output_dir, model_name)
        )

        _check_prediction_file(pred_df, config, model_name)
        _check_backtest_outputs(
            daily_returns=daily_returns,
            daily_nav=daily_nav,
            daily_weights=daily_weights,
            daily_turnover=daily_turnover,
            config=config,
            model_name=model_name,
        )

        alignment_samples = _check_return_alignment(
            pred_df=pred_df,
            returns_df=returns_df,
            config=config,
            model_name=model_name,
        )

        top_n_summary = _summarize_saved_backtest(
            daily_returns=daily_returns,
            daily_turnover=daily_turnover,
            config=config,
        )
        _assert_summary_matches_comparison(
            model_name=model_name,
            summary=top_n_summary,
            comparison_df=comparison_df,
        )

        ic_summary = prediction_ic_summary(
            pred_df=pred_df,
            date_col=config.date_col,
            y_true_col="y_true",
            y_pred_col="y_pred",
            min_obs=50,
        )

        equal_summary = _run_equal_all_baseline(
            pred_df=pred_df,
            returns_df=returns_df,
            config=config,
        )

        model_check_rows.append(
            {
                "model": model_name,
                "prediction_rows": len(pred_df),
                "backtest_periods": len(daily_returns),
                "prediction_start": pred_df[config.date_col].min(),
                "prediction_end": pred_df[config.date_col].max(),
                "return_start": daily_returns.index.min(),
                "return_end": daily_returns.index.max(),
                "mean_turnover": float(daily_turnover.mean()),
            }
        )

        ic_row = {"model": model_name}
        ic_row.update(ic_summary)
        ic_rows.append(ic_row)

        baseline_rows.append(
            {
                "model": model_name,
                "top_n_sharpe": top_n_summary["sharpe_ratio"],
                "equal_all_sharpe": equal_summary["sharpe_ratio"],
                "top_n_final_nav": top_n_summary["final_nav"],
                "equal_all_final_nav": equal_summary["final_nav"],
                "top_n_annualized_return": top_n_summary["annualized_return"],
                "equal_all_annualized_return": equal_summary["annualized_return"],
                "top_n_max_drawdown": top_n_summary["max_drawdown"],
                "equal_all_max_drawdown": equal_summary["max_drawdown"],
                "top_n_mean_turnover": top_n_summary["mean_turnover"],
                "equal_all_mean_turnover": equal_summary["mean_turnover"],
            }
        )

        alignment_rows.extend(alignment_samples)

        print(f"{model_name} checks passed.")
        print("Top-N summary:", top_n_summary)
        print("IC summary:", ic_summary)
        print("Equal-all summary:", equal_summary)

    sanity_report_path = _write_sanity_report(
        report_path=Path("reports/experiment_001_sanity_checks.md"),
        model_check_rows=model_check_rows,
        ic_rows=ic_rows,
        baseline_rows=baseline_rows,
        alignment_rows=alignment_rows,
        comparison_df=comparison_df,
    )

    print("\nAll multi-model sanity checks passed.")
    print(f"Sanity report saved to: {sanity_report_path}")


if __name__ == "__main__":
    main()
