"""
Run a single-model end-to-end experiment.

Phase 2A goal:
raw data -> preprocess -> dataset -> train -> predict -> backtest -> report
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtest
from src.backtest.portfolio import PortfolioConfig
from src.data.dataset_builder import PanelDatasetBuilder
from src.data.loader import load_panel_data
from src.data.preprocess import PreprocessConfig, preprocess_panel_data
from src.models.lightgbm_model import LightGBMConfig, build_lightgbm_model
from src.predict.generate_predictions import generate_predictions, save_predictions
from src.train.train_tabular import train_tabular_model
from src.utils.seed import set_seed


DEFAULT_FEATURE_COLS = (
    "SIZE",
    "SIZENL",
    "LIQUIDITY",
    "BETA",
    "RESVOL",
    "MOMENTUM",
    "LEVERAGE",
    "VALUE",
    "EARNYLD",
    "GROWTH",
    "LTREV",
    "STREV",
)


@dataclass
class ExperimentConfig:
    data_path: str = "dataset/input/A_2010_2020.parquet"

    date_col: str = "time"
    stock_col: str = "stock_id"
    target_col: str = "1d_next_raw"
    return_col: str = "return_1d"

    feature_cols: Sequence[str] = DEFAULT_FEATURE_COLS
    meta_cols: Sequence[str] = ("industry_code", "price")

    train_end: str = ""
    valid_end: str = ""

    seed: int = 42
    top_n: int = 50
    periods_per_year: int = 252

    output_dir: str = "dataset/output/experiment_001"
    report_path: str = "reports/experiment_001.md"


def _split_clean_df_7_1_2(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """
    Split a cleaned panel DataFrame into strict 7/1/2 chronological date slices.

    The split is based on unique dates, not row counts, to avoid splitting the
    same trading day across train/valid/test.
    """
    required_cols = [date_col, stock_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df missing required columns: {missing_cols}")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="raise")
    work = work.sort_values([date_col, stock_col], kind="mergesort").reset_index(
        drop=True
    )

    unique_dates = pd.DatetimeIndex(work[date_col].drop_duplicates()).sort_values()
    n_dates = len(unique_dates)

    if n_dates < 3:
        raise ValueError(f"Need at least 3 unique dates for 7/1/2 split, got {n_dates}")

    train_end_idx = int(n_dates * 0.7)
    valid_end_idx = int(n_dates * 0.8)

    if train_end_idx <= 0 or valid_end_idx <= train_end_idx or valid_end_idx >= n_dates:
        raise ValueError(
            "Unable to derive a non-empty 7/1/2 date split. "
            f"n_dates={n_dates}, train_end_idx={train_end_idx}, "
            f"valid_end_idx={valid_end_idx}"
        )

    train_dates = unique_dates[:train_end_idx]
    valid_dates = unique_dates[train_end_idx:valid_end_idx]
    test_dates = unique_dates[valid_end_idx:]

    train_df = work[work[date_col].isin(train_dates)].copy().reset_index(drop=True)
    valid_df = work[work[date_col].isin(valid_dates)].copy().reset_index(drop=True)
    test_df = work[work[date_col].isin(test_dates)].copy().reset_index(drop=True)

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "Strict 7/1/2 date split produced an empty subset. "
            f"train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}"
        )

    train_end = str(pd.Timestamp(train_dates[-1]))
    valid_end = str(pd.Timestamp(valid_dates[-1]))

    return train_df, valid_df, test_df, train_end, valid_end


def _load_experiment_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load the experiment input data.

    If the aggregate parquet is unreadable, fall back to split parquet files in
    the same directory.
    """
    try:
        return load_panel_data(data_path)
    except OSError as exc:
        error_message = str(exc)

        if "Couldn't deserialize thrift" not in error_message:
            raise

        data_file = Path(data_path)
        required_fallback_paths = [
            data_file.with_name("df_response_daily_train.parquet"),
            data_file.with_name("df_response_daily_validate.parquet"),
        ]
        optional_fallback_paths = [
            data_file.with_name("df_response_daily_test.parquet"),
        ]

        missing_paths = [path for path in required_fallback_paths if not path.exists()]
        if missing_paths:
            missing_text = ", ".join(str(path) for path in missing_paths)
            raise ValueError(
                "Failed to read aggregate parquet and fallback files are missing: "
                f"{missing_text}"
            ) from exc

        fallback_paths = required_fallback_paths + [
            path for path in optional_fallback_paths if path.exists()
        ]

        frames = [load_panel_data(path) for path in fallback_paths]

        return pd.concat(frames, ignore_index=True)


def build_returns_frame_from_next_target(
    df: pd.DataFrame,
    date_col: str,
    stock_col: str,
    target_col: str,
    return_col: str,
) -> pd.DataFrame:
    """
    Convert a next-period target column into an engine-compatible return frame.

    If target_col at signal date t means the return from t to the next market
    trading date, this function assigns that value to the next global date.
    """
    required_cols = [date_col, stock_col, target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"df missing required columns: {missing_cols}")

    work = df[required_cols].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="raise")
    work = work.sort_values([date_col, stock_col], kind="mergesort").reset_index(
        drop=True
    )

    unique_dates = pd.DatetimeIndex(work[date_col].drop_duplicates()).sort_values()

    next_date_map = {
        unique_dates[i]: unique_dates[i + 1]
        for i in range(len(unique_dates) - 1)
    }

    returns_df = work.copy()
    returns_df[date_col] = returns_df[date_col].map(next_date_map)
    returns_df = returns_df.loc[returns_df[date_col].notna()].copy()

    returns_df = returns_df.rename(columns={target_col: return_col})
    returns_df = returns_df[[date_col, stock_col, return_col]]

    returns_df[stock_col] = returns_df[stock_col].astype(str)
    returns_df[return_col] = returns_df[return_col].astype(float)

    finite_mask = np.isfinite(returns_df[return_col].to_numpy())
    returns_df = returns_df.loc[finite_mask].reset_index(drop=True)

    if returns_df.duplicated(subset=[date_col, stock_col]).any():
        raise ValueError(
            f"returns_df contains duplicated {date_col}-{stock_col} rows"
        )

    return returns_df


def save_markdown_report(
    report_path: str | Path,
    config: ExperimentConfig,
    preprocess_report: dict,
    train_summary: dict,
    backtest_result: dict,
    pred_df: pd.DataFrame,
    split_sizes: dict[str, int],
) -> Path:
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    summary = backtest_result["summary"]

    lines = [
        "# Experiment 001: LightGBM End-to-End Backtest",
        "",
        "## Config",
        "",
        f"- data_path: `{config.data_path}`",
        f"- date_col: `{config.date_col}`",
        f"- stock_col: `{config.stock_col}`",
        f"- target_col: `{config.target_col}`",
        f"- train_end: `{config.train_end}`",
        f"- valid_end: `{config.valid_end}`",
        f"- top_n: `{config.top_n}`",
        "",
        "## Data Summary",
        "",
        f"- preprocess input rows: {preprocess_report.get('input_rows')}",
        f"- preprocess output rows: {preprocess_report.get('output_rows')}",
        f"- train rows: {split_sizes['train_rows']}",
        f"- valid rows: {split_sizes['valid_rows']}",
        f"- test rows: {split_sizes['test_rows']}",
        f"- prediction rows: {len(pred_df)}",
        (
            f"- prediction date range: "
            f"{pred_df[config.date_col].min()} to {pred_df[config.date_col].max()}"
        ),
        "",
        "## Training Summary",
        "",
    ]

    for key, value in train_summary.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Backtest Summary",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )

    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.6f} |")
        else:
            lines.append(f"| {key} | {value} |")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    return report_path


def main() -> None:
    config = ExperimentConfig()

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_df, valid_df, test_df, config.train_end, config.valid_end = (
        _split_clean_df_7_1_2(
            clean_df,
            config.date_col,
            config.stock_col,
        )
    )

    split_sizes = {
        "train_rows": len(train_df),
        "valid_rows": len(valid_df),
        "test_rows": len(test_df),
    }

    builder = PanelDatasetBuilder(
        feature_cols=list(config.feature_cols),
        target_col=config.target_col,
        date_col=config.date_col,
        stock_col=config.stock_col,
        seq_len=60,
        meta_cols=list(config.meta_cols),
    )

    train_data = builder.build_tabular_dataset(train_df)
    valid_data = builder.build_tabular_dataset(valid_df)
    test_data = builder.build_tabular_dataset(test_df)

    model = build_lightgbm_model(
        LightGBMConfig(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            early_stopping_rounds=50,
            verbose_eval=False,
            random_state=config.seed,
        )
    )

    train_summary = train_tabular_model(
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        output_dir=output_dir / "models",
    )

    pred_df = generate_predictions(
        model=model,
        dataset=test_data,
        model_name="lightgbm",
        required_meta_cols=(config.date_col, config.stock_col),
    )

    prediction_path = save_predictions(
        pred_df=pred_df,
        output_path=output_dir / "predictions_lightgbm.parquet",
    )

    returns_df = build_returns_frame_from_next_target(
        df=clean_df,
        date_col=config.date_col,
        stock_col=config.stock_col,
        target_col=config.target_col,
        return_col=config.return_col,
    )

    portfolio_config = PortfolioConfig(
        strategy="top_n",
        top_n=config.top_n,
        pred_col="y_pred",
        stock_col=config.stock_col,
    )

    backtest_result = run_backtest(
        pred_df=pred_df,
        returns_df=returns_df,
        portfolio_config=portfolio_config,
        return_col=config.return_col,
        date_col=config.date_col,
        stock_col=config.stock_col,
        periods_per_year=config.periods_per_year,
    )

    backtest_result["daily_returns"].to_csv(
        output_dir / "daily_returns_lightgbm.csv",
        header=True,
    )

    backtest_result["daily_nav"].to_csv(
        output_dir / "daily_nav_lightgbm.csv",
        header=True,
    )

    backtest_result["daily_weights"].to_csv(
        output_dir / "daily_weights_lightgbm.csv",
    )

    backtest_result["daily_turnover"].to_csv(
        output_dir / "daily_turnover_lightgbm.csv",
        header=True,
    )

    report_path = save_markdown_report(
        report_path=config.report_path,
        config=config,
        preprocess_report=preprocess_result.report,
        train_summary=train_summary,
        backtest_result=backtest_result,
        pred_df=pred_df,
        split_sizes=split_sizes,
    )

    print("Experiment completed.")
    print(f"Predictions saved to: {prediction_path}")
    print(f"Report saved to: {report_path}")
    print("Split sizes:")
    print(split_sizes)
    print("Backtest summary:")
    print(backtest_result["summary"])


if __name__ == "__main__":
    main()