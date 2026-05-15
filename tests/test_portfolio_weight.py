"""
Tests for portfolio weight construction.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.portfolio import PortfolioConfig, build_portfolio_weights


def _make_prediction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2020-01-01"] * 5,
            "stock_id": ["A", "B", "C", "D", "E"],
            "y_pred": [0.03, -0.01, 0.02, 0.05, 0.00],
        }
    )


def test_equal_all_weights_sum_to_one() -> None:
    pred_df = _make_prediction_df()

    config = PortfolioConfig(
        strategy="equal_all",
        top_n=3,
        pred_col="y_pred",
        stock_col="stock_id",
    )

    weights = build_portfolio_weights(pred_df, config)

    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert len(weights) == len(pred_df)


def test_equal_all_assigns_equal_weights_to_all_stocks() -> None:
    pred_df = _make_prediction_df()

    config = PortfolioConfig(
        strategy="equal_all",
        stock_col="stock_id",
    )

    weights = build_portfolio_weights(pred_df, config)

    expected_weight = 1.0 / len(pred_df)

    assert set(weights.index) == set(pred_df["stock_id"])
    assert np.allclose(weights.to_numpy(), expected_weight)


def test_top_n_weights_sum_to_one() -> None:
    pred_df = _make_prediction_df()

    config = PortfolioConfig(
        strategy="top_n",
        top_n=3,
        pred_col="y_pred",
        stock_col="stock_id",
    )

    weights = build_portfolio_weights(pred_df, config)

    assert isinstance(weights, pd.Series)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0.0).all()
    assert len(weights) == 3


def test_top_n_selects_highest_predictions() -> None:
    pred_df = _make_prediction_df()

    config = PortfolioConfig(
        strategy="top_n",
        top_n=3,
        pred_col="y_pred",
        stock_col="stock_id",
    )

    weights = build_portfolio_weights(pred_df, config)

    expected_selected = {"D", "A", "C"}

    assert set(weights.index) == expected_selected
    assert (weights.loc[list(expected_selected)] > 0.0).all()


def test_top_n_assigns_equal_weights_to_selected_stocks() -> None:
    pred_df = _make_prediction_df()

    config = PortfolioConfig(
        strategy="top_n",
        top_n=3,
        pred_col="y_pred",
        stock_col="stock_id",
    )

    weights = build_portfolio_weights(pred_df, config)

    assert np.allclose(weights.to_numpy(), 1.0 / 3.0)


def test_top_n_larger_than_universe_selects_all_stocks() -> None:
    pred_df = _make_prediction_df()

    config = PortfolioConfig(
        strategy="top_n",
        top_n=10,
        pred_col="y_pred",
        stock_col="stock_id",
    )

    weights = build_portfolio_weights(pred_df, config)

    assert len(weights) == len(pred_df)
    assert weights.sum() == pytest.approx(1.0)
    assert set(weights.index) == set(pred_df["stock_id"])


def test_duplicate_stock_id_raises_error() -> None:
    pred_df = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "stock_id": ["A", "A", "B"],
            "y_pred": [0.03, 0.02, 0.01],
        }
    )

    config = PortfolioConfig(
        strategy="top_n",
        top_n=2,
        pred_col="y_pred",
        stock_col="stock_id",
    )

    with pytest.raises(ValueError, match="duplicated"):
        build_portfolio_weights(pred_df, config)


def test_top_n_missing_prediction_column_raises_error() -> None:
    pred_df = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "stock_id": ["A", "B", "C"],
        }
    )

    config = PortfolioConfig(
        strategy="top_n",
        top_n=2,
        pred_col="y_pred",
        stock_col="stock_id",
    )

    with pytest.raises(ValueError, match="prediction column"):
        build_portfolio_weights(pred_df, config)


def test_equal_all_does_not_require_prediction_column() -> None:
    pred_df = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "stock_id": ["A", "B", "C"],
        }
    )

    config = PortfolioConfig(
        strategy="equal_all",
        stock_col="stock_id",
    )

    weights = build_portfolio_weights(pred_df, config)

    assert len(weights) == 3
    assert weights.sum() == pytest.approx(1.0)
    assert set(weights.index) == {"A", "B", "C"}


def test_missing_stock_column_raises_error() -> None:
    pred_df = pd.DataFrame(
        {
            "date": ["2020-01-01"] * 3,
            "y_pred": [0.03, 0.02, 0.01],
        }
    )

    config = PortfolioConfig(
        strategy="top_n",
        top_n=2,
        pred_col="y_pred",
        stock_col="stock_id",
    )

    with pytest.raises(ValueError, match="stock column"):
        build_portfolio_weights(pred_df, config)


def test_invalid_strategy_raises_error() -> None:
    with pytest.raises(ValueError, match="strategy"):
        PortfolioConfig(strategy="rank")


def test_invalid_top_n_raises_error() -> None:
    with pytest.raises(ValueError, match="top_n"):
        PortfolioConfig(strategy="top_n", top_n=0)