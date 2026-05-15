"""
Portfolio construction and management utilities.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

@dataclass
class PortfolioConfig:
    """
    Configuration for portfolio construction.

    strategy : str
        Portfolio construction strategy. Supported values:
        - "equal_all": equal weight over all available stocks.
        - "top_n": select top N stocks by prediction and equal weight them.
    top_n : int
        Number of stocks selected for the top_n strategy.
    pred_col : str
        Prediction column used for ranking.
    stock_col : str
        Stock identifier column.
    """
    strategy: str = "top_n"
    top_n: int = 50
    pred_col: str = "y_pred"
    stock_col: str = "stock_id"

    def __post_init__(self) -> None:
        if self.strategy not in {"equal_all", "top_n"}:
            raise ValueError(f"unsupported strategy: {self.strategy}")
        if self.top_n <= 0:
            raise ValueError("top_n must be a positive integer")
        if not self.pred_col:
            raise ValueError("pred_col must be a non-empty string")
        if not self.stock_col:
            raise ValueError("stock_col must be a non-empty string")
        
def _validate_prediction_frame(pred_df: pd.DataFrame, config: PortfolioConfig) -> None:
    """
    Validate the prediction DataFrame against the portfolio configuration.
    """
    if not isinstance(pred_df, pd.DataFrame):
        raise TypeError(f"pred_df must be a DataFrame, got {type(pred_df).__name__}")

    if pred_df.empty:
        raise ValueError("pred_df must not be empty")

    if config.stock_col not in pred_df.columns:
        raise ValueError(f"pred_df missing stock column: {config.stock_col}")

    if pred_df[config.stock_col].isna().any():
        raise ValueError(f"{config.stock_col} contains missing values")

    if pred_df[config.stock_col].duplicated().any():
        raise ValueError(f"{config.stock_col} contains duplicated stocks")

    if config.strategy == "top_n":
        if config.pred_col not in pred_df.columns:
            raise ValueError(f"pred_df missing prediction column: {config.pred_col}")

        pred_values = pred_df[config.pred_col].to_numpy(dtype=float)

        if not np.isfinite(pred_values).all():
            raise ValueError(f"{config.pred_col} contains NaN or Inf values")
        
def _equal_weight(stock_ids: pd.Series) -> pd.Series:
    """
    Assign equal weights to the given stock IDs.
    """
    n_stocks = len(stock_ids)

    if n_stocks == 0:
        raise ValueError("no stocks to assign weights to")
    
    weight = 1.0 / n_stocks

    return pd.Series(
        data=weight,
        index=stock_ids,
        name="weight",
        dtype=float
    )

def _build_equal_all_weights(pred_df: pd.DataFrame, config: PortfolioConfig,) -> pd.Series:
    """
    Build equal weights over the full available stock universe.
    """
    stock_ids = pred_df[config.stock_col].reset_index(drop=True)

    return _equal_weight(stock_ids)

def _build_top_n_weights(pred_df: pd.DataFrame, config: PortfolioConfig) -> pd.Series:
    """
    Select top N stocks by prediction and assign equal weights to them.
    """
    sorted_df = pred_df.sort_values(
        by=[config.pred_col, config.stock_col],
        ascending=[False, True],
        kind="mergesort"
    )

    n_select = min(config.top_n, len(sorted_df))
    selected = sorted_df.head(n_select)

    stock_ids = selected[config.stock_col].reset_index(drop=True)

    return _equal_weight(stock_ids)

def build_portfolio_weights(pred_df: pd.DataFrame, config: PortfolioConfig | None = None,) -> pd.Series:
    """
    Build portfolio weights from one-period prediction DataFrame.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Prediction DataFrame for a single date. It should contain at least
        stock_col, and pred_col when using the top_n strategy.
    config : PortfolioConfig, optional
        Portfolio construction configuration.

    Returns
    -------
    pd.Series
        Portfolio weights indexed by stock identifier.
    """
    if config is None:
        config = PortfolioConfig()

    _validate_prediction_frame(pred_df, config)

    if config.strategy == "equal_all":
        weights = _build_equal_all_weights(pred_df, config)
    elif config.strategy == "top_n":
        weights = _build_top_n_weights(pred_df, config)
    else:
        raise ValueError(f"unsupported strategy: {config.strategy}")

    weight_sum = float(weights.sum())

    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"weights must sum to 1.0, got {weight_sum}")

    if (weights < 0).any():
        raise ValueError("weights must be non-negative")

    return weights

if __name__ == "__main__":
    sample_pred = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 5,
            "stock_id": ["A", "B", "C", "D", "E"],
            "y_pred": [0.03, -0.01, 0.02, 0.05, 0.00],
        }
    )

    equal_config = PortfolioConfig(strategy="equal_all")
    equal_weights = build_portfolio_weights(sample_pred, equal_config)

    print("Equal-all weights:")
    print(equal_weights)
    print("Sum:", equal_weights.sum())

    top_n_config = PortfolioConfig(strategy="top_n", top_n=3)
    top_n_weights = build_portfolio_weights(sample_pred, top_n_config)

    print("\nTop-N weights:")
    print(top_n_weights)
    print("Sum:", top_n_weights.sum())