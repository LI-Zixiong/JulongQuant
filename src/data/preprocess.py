from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

@dataclass
class PreprocessConfig:
    """
    Configuration for basic panel data preprocessing.

    V0 only performs basic cleaning and validation. It does not standardize
    factors, winsorize features, neutralize by industry, construct targets,
    or split train / valid / test data.
    
        Notes
        -----
        - `duplicate_policy`: V0 only supports "raise". If duplicates by
            (stock_id, date) are present, preprocessing will raise an error.
    """

    date_col: str = "date"
    stock_col: str = "stock_id"

    feature_cols: Optional[Sequence[str]] = None
    target_col: Optional[str] = None
    meta_cols: Optional[Sequence[str]] = None

    replace_inf_with_nan: bool = True

    drop_rows_with_missing_keys: bool = True
    drop_rows_with_missing_features: bool = False
    drop_rows_with_missing_target:bool = False

    duplicate_policy: str = 'raise'

    sort_values: bool = True

@dataclass
class PreprocessResult:
    """
    Result returned by preprocess_panel_data.

    Attributes
    ----------
    df:
        Cleaned panel DataFrame.

    report:
        Lightweight cleaning report for debugging and data-quality checks.
    """

    df: pd.DataFrame
    report: Dict[str, Any]

def _deduplicate_preserve_order(values: Sequence[str]) -> List[str]:
    """
    Remove duplicated values while preserving their original order.
    """
    
    seen = set()
    result = []

    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

def _validate_config(config: PreprocessConfig) -> None:
    """
    Validate the preprocess config.
    """

    if not isinstance(config.date_col, str) or config.date_col == "":
        raise ValueError("date_col must be a non-empty string")

    if not isinstance(config.stock_col, str) or config.stock_col == "":
        raise ValueError("stock_col must be a non-empty string")

    if config.target_col is not None:
        if not isinstance(config.target_col, str) or config.target_col == "":
            raise ValueError("target_col must be None or a non-empty string")

    if config.duplicate_policy != "raise":
        raise ValueError("V0 only supports duplicate_policy='raise'")

    if config.feature_cols is not None:
        feature_cols = list(config.feature_cols)
        if len(feature_cols) == 0:
            raise ValueError("feature_cols must be None or a non-empty sequence")
        if len(set(feature_cols)) != len(feature_cols):
            raise ValueError("feature_cols contains duplicated column names")

    if config.meta_cols is not None:
        meta_cols = list(config.meta_cols)
        if len(set(meta_cols)) != len(meta_cols):
            raise ValueError("meta_cols contains duplicated column names")
        
def _required_columns(config: PreprocessConfig) -> List[str]:
    """
    Get the list of required columns based on the config.
    """

    cols: List[str] = [config.date_col, config.stock_col]

    if config.feature_cols is not None:
        cols.extend(list(config.feature_cols))

    if config.target_col is not None:
        cols.append(config.target_col)

    if config.meta_cols is not None:
        cols.extend(list(config.meta_cols))

    return _deduplicate_preserve_order(cols)

def _validate_columns(df: pd.DataFrame, config: PreprocessConfig) -> None:
    """
    Check whether the input DataFrame contains all required columns.
    """

    missing_cols = [col for col in _required_columns(config) if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")
    
def _check_duplicate_stock_date(
    df: pd.DataFrame,
    config: PreprocessConfig,
) -> int:
    """
    Check duplicated stock-date rows.

    V0 only supports duplicate_policy='raise'. If duplicated stock-date rows are
    found, this function raises an error and reports how many duplicated rows
    exist.

    Returns
    -------
    int
        Number of duplicated rows.
    """

    duplicated_mask = df.duplicated(
        subset=[config.stock_col, config.date_col],
        keep=False,
    )
    duplicated_count = int(duplicated_mask.sum())

    if duplicated_count > 0:
        sample_duplicates = (
            df.loc[duplicated_mask, [config.stock_col, config.date_col]]
            .head(5)
            .to_dict("records")
        )

        raise ValueError(
            "Found duplicated stock-date rows. "
            f"Duplicated row count: {duplicated_count}. "
            "V0 only supports duplicate_policy='raise'. "
            "Please resolve duplicates before model dataset construction. "
            f"Sample duplicates: {sample_duplicates}"
        )

    return duplicated_count

def preprocess_panel_data(
    df: pd.DataFrame,
    config: PreprocessConfig,
) -> PreprocessResult:
    """
    Basic V0 preprocessing for stock panel data.

    This function only performs structural cleaning and validation:
        - validate required columns
        - convert date column to datetime
        - convert stock id column to string
        - optionally drop rows with missing keys / features / target
        - optionally replace Inf / -Inf with NaN
        - check duplicated stock-date rows
        - optionally sort by stock and date

    It does not perform factor standardization, winsorization, industry
    neutralization, target construction, or train / valid / test splitting.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")

    _validate_config(config)
    _validate_columns(df, config)

    cleaned = df.copy()

    report: Dict[str, Any] = {
        "input_rows": int(len(cleaned)),
        "output_rows": None,
        "dropped_missing_key_rows": 0,
        "dropped_missing_feature_rows": 0,
        "dropped_missing_target_rows": 0,
        "duplicate_rows": 0,
        "inf_values_replaced": 0,
    }

    # Convert date column first.
    # - None/NaN values are converted to NaT and can be removed by the
    #   missing-key removal logic below.
    # - Invalid / unparseable date strings will raise immediately
    #   (errors="raise"), which is the V0 behavior.
    cleaned[config.date_col] = pd.to_datetime(cleaned[config.date_col], errors="raise")

    # Drop rows with missing panel keys before converting stock ids to string.
    if config.drop_rows_with_missing_keys:
        missing_date = cleaned[config.date_col].isna()

        stock_as_str = cleaned[config.stock_col].fillna("").astype(str).str.strip()
        missing_stock = cleaned[config.stock_col].isna() | stock_as_str.eq("")

        missing_key_mask = missing_date | missing_stock

        report["dropped_missing_key_rows"] = int(missing_key_mask.sum())
        cleaned = cleaned.loc[~missing_key_mask].copy()

    # Use string stock ids consistently after missing-key handling.
    cleaned[config.stock_col] = cleaned[config.stock_col].astype(str)

    # Replace Inf / -Inf with NaN. Missing-value decisions remain explicit.
    if config.replace_inf_with_nan:
        numeric_cols = list(cleaned.select_dtypes(include=[np.number]).columns)
        if numeric_cols:
            numeric_values = cleaned[numeric_cols].to_numpy(dtype=np.float64, copy=True)
            inf_count = int(np.isinf(numeric_values).sum())
            report["inf_values_replaced"] = inf_count
            cleaned[numeric_cols] = cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Optionally drop rows with missing features.
    if config.drop_rows_with_missing_features and config.feature_cols is not None:
        missing_feature_mask = cleaned[list(config.feature_cols)].isna().any(axis=1)
        report["dropped_missing_feature_rows"] = int(missing_feature_mask.sum())
        cleaned = cleaned.loc[~missing_feature_mask].copy()

    # Optionally drop rows with missing target.
    if config.drop_rows_with_missing_target and config.target_col is not None:
        missing_target_mask = cleaned[config.target_col].isna()
        report["dropped_missing_target_rows"] = int(missing_target_mask.sum())
        cleaned = cleaned.loc[~missing_target_mask].copy()

    # Duplicated stock-date rows are treated as data-quality errors in V0.
    report["duplicate_rows"] = _check_duplicate_stock_date(cleaned, config)

    if config.sort_values:
        cleaned = cleaned.sort_values(
            [config.stock_col, config.date_col],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        cleaned = cleaned.reset_index(drop=True)

    report["output_rows"] = int(len(cleaned))

    return PreprocessResult(df=cleaned, report=report)


if __name__ == "__main__":
    # Minimal smoke test for V0 preprocessing.
    raw_df = pd.DataFrame(
        {
            "date": ["2020-01-02", "2020-01-01", "2020-01-03", None],
            "stock_id": ["B", "A", "A", "C"],
            "factor_1": [1.0, np.inf, 3.0, 4.0],
            "factor_2": [0.5, 0.2, np.nan, 0.1],
            "target": [0.01, 0.02, np.nan, 0.04],
            "industry_code": ["IND_2", "IND_1", "IND_1", "IND_3"],
        }
    )

    config = PreprocessConfig(
        date_col="date",
        stock_col="stock_id",
        feature_cols=["factor_1", "factor_2"],
        target_col="target",
        meta_cols=["industry_code"],
        replace_inf_with_nan=True,
        drop_rows_with_missing_keys=True,
        drop_rows_with_missing_features=False,
        drop_rows_with_missing_target=False,
        duplicate_policy="raise",
        sort_values=True,
    )

    result = preprocess_panel_data(raw_df, config)

    print("Cleaned DataFrame:")
    print(result.df)

    print("Report:")
    print(result.report)

    print("Date dtype:", result.df["date"].dtype)
    print("Stock dtype:", result.df["stock_id"].dtype)
