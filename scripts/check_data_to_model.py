"""
Minimal smoke test for the data-to-model pipeline.

Goal:
    loader -> preprocess V0 -> dataset_builder -> LightGBM fit/predict

This script is for checking whether the first pipeline version can run end-to-end.
It is not a formal training script and should not be used for final experiments.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project import setup
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_panel_data
from src.data.preprocess import PreprocessConfig, preprocess_panel_data
from src.data.dataset_builder import PanelDatasetBuilder
from src.models.lightgbm_model import LightGBMConfig, build_lightgbm_model


# ---------------------------------------------------------------------
# User settings: edit these first
# ---------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "dataset" / "input" / "df_response_daily_train.parquet"

DATE_COL = "time"
STOCK_COL = "stock_id"
TARGET_COL = "5d_next_styAdj"

FEATURE_COLS = [
    "SIZE",
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
    "price",
]

META_COLS = [
    "industry_code",
    "st_status",
    "price",
]

# For this smoke test, None means automatically infer split dates from data.
TRAIN_END = None
VALID_END = None


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def infer_date_splits(
    df: pd.DataFrame,
    date_col: str,
    train_frac: float = 0.6,
    valid_frac: float = 0.8,
) -> tuple[str, str]:
    """
    Infer train_end and valid_end from sorted unique dates.

    This is only for smoke testing. Formal experiments should use fixed dates.
    """

    dates = pd.Series(pd.to_datetime(df[date_col], errors="raise")).dropna().sort_values().unique()

    if len(dates) < 5:
        raise ValueError(
            f"Need at least 5 unique dates to infer splits, got {len(dates)}"
        )

    train_idx = max(0, min(int(len(dates) * train_frac) - 1, len(dates) - 3))
    valid_idx = max(train_idx + 1, min(int(len(dates) * valid_frac) - 1, len(dates) - 2))

    train_end = pd.Timestamp(dates[train_idx]).strftime("%Y-%m-%d")
    valid_end = pd.Timestamp(dates[valid_idx]).strftime("%Y-%m-%d")

    return train_end, valid_end


def print_dataset_summary(name: str, data) -> None:
    print(f"{name} X shape: {data.X.shape}")
    print(f"{name} y shape: {data.y.shape}")
    print(f"{name} meta shape: {data.meta.shape}")
    print(f"{name} date range: {data.meta[DATE_COL].min()} -> {data.meta[DATE_COL].max()}")
    print(f"{name} stock count: {data.meta[STOCK_COL].nunique()}")


# ---------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------
def main() -> None:
    print("Step 1. Load raw data")
    df_raw = load_panel_data(DATA_PATH)
    print("Raw df shape:", df_raw.shape)
    print("Raw columns:", list(df_raw.columns))

    if DATE_COL not in df_raw.columns:
        if DATE_COL in df_raw.index.names:
            df_raw = df_raw.reset_index()
        else:
            raise ValueError(
                f"DATE_COL='{DATE_COL}' is neither a column nor an index name. "
                f"Columns: {list(df_raw.columns)}, index names: {df_raw.index.names}"
            )

    missing_features = [col for col in FEATURE_COLS if col not in df_raw.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if TARGET_COL not in df_raw.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    available_meta_cols = [col for col in META_COLS if col in df_raw.columns]
    missing_meta_cols = [col for col in META_COLS if col not in df_raw.columns]
    if missing_meta_cols:
        print("Warning: missing meta columns ignored:", missing_meta_cols)

    print("Step 2. Preprocess data")
    preprocess_config = PreprocessConfig(
        date_col=DATE_COL,
        stock_col=STOCK_COL,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        meta_cols=available_meta_cols,
        replace_inf_with_nan=True,
        drop_rows_with_missing_keys=True,
        drop_rows_with_missing_features=False,
        drop_rows_with_missing_target=False,
        duplicate_policy="raise",
        sort_values=True,
    )

    preprocess_result = preprocess_panel_data(df_raw, preprocess_config)
    df_clean = preprocess_result.df
    print("Clean df shape:", df_clean.shape)
    print("Preprocess report:", preprocess_result.report)

    print("Step 3. Build tabular train / valid / test datasets")
    train_end, valid_end = TRAIN_END, VALID_END
    if train_end is None or valid_end is None:
        train_end, valid_end = infer_date_splits(df_clean, DATE_COL)

    print("train_end:", train_end)
    print("valid_end:", valid_end)

    builder = PanelDatasetBuilder(
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        date_col=DATE_COL,
        stock_col=STOCK_COL,
        seq_len=60,
        meta_cols=available_meta_cols,
    )

    train_data, valid_data, test_data = builder.build_tabular_splits(
        df=df_clean,
        train_end=train_end,
        valid_end=valid_end,
    )

    print_dataset_summary("Train", train_data)
    print_dataset_summary("Valid", valid_data)
    print_dataset_summary("Test", test_data)

    print("Step 4. Fit a small LightGBM model")
    model_config = LightGBMConfig(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        early_stopping_rounds=10,
        verbose_eval=False,
    )
    model = build_lightgbm_model(model_config)

    model.fit(
        train_data.X,
        train_data.y,
        valid_data.X,
        valid_data.y,
    )

    print("Step 5. Predict on test data")
    preds = model.predict(test_data.X)
    print("Prediction shape:", preds.shape)
    print("First 5 predictions:", preds[:5])

    pred_df = test_data.meta.copy()
    pred_df["y_true"] = test_data.y
    pred_df["y_pred"] = preds

    print("Prediction DataFrame preview:")
    print(pred_df.head())

    output_dir = PROJECT_ROOT / "dataset" / "output" / "smoke_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lightgbm_smoke_predictions.parquet"
    pred_df.to_parquet(output_path, index=False)
    print("Saved smoke predictions to:", output_path)


if __name__ == "__main__":
    main()
