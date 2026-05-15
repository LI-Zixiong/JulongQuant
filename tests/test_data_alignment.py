"""
Tests for BuiltDataset data alignment.

These tests make sure X, y, and meta remain aligned after dataset construction,
subsetting, and date-based splitting.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builder import BuiltDataset, PanelDatasetBuilder


def _make_panel_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    stock_ids = ["A", "B", "C"]

    rows = []

    for date_idx, date in enumerate(dates):
        for stock_idx, stock_id in enumerate(stock_ids):
            row_id = date_idx * 10 + stock_idx

            rows.append(
                {
                    "date": date,
                    "stock_id": stock_id,
                    "factor_1": float(row_id),
                    "factor_2": float(row_id) + 0.5,
                    "target": float(row_id) + 1.0,
                    "row_id": float(row_id),
                    "industry_code": f"IND_{stock_idx}",
                }
            )

    return pd.DataFrame(rows)


def _make_builder(seq_len: int = 3) -> PanelDatasetBuilder:
    return PanelDatasetBuilder(
        feature_cols=["factor_1", "factor_2"],
        target_col="target",
        date_col="date",
        stock_col="stock_id",
        seq_len=seq_len,
        meta_cols=["row_id", "industry_code"],
    )


def _assert_tabular_alignment(dataset: BuiltDataset) -> None:
    assert dataset.X.shape[0] == dataset.y.shape[0] == len(dataset.meta)

    row_id = dataset.meta["row_id"].to_numpy(dtype=np.float32)

    np.testing.assert_allclose(dataset.X[:, 0], row_id)
    np.testing.assert_allclose(dataset.X[:, 1], row_id + 0.5)
    np.testing.assert_allclose(dataset.y, row_id + 1.0)


def _assert_sequence_alignment(dataset: BuiltDataset, seq_len: int) -> None:
    assert dataset.X.shape[0] == dataset.y.shape[0] == len(dataset.meta)
    assert dataset.X.shape[1] == seq_len

    row_id = dataset.meta["row_id"].to_numpy(dtype=np.float32)

    np.testing.assert_allclose(dataset.X[:, -1, 0], row_id)
    np.testing.assert_allclose(dataset.X[:, -1, 1], row_id + 0.5)
    np.testing.assert_allclose(dataset.y, row_id + 1.0)


def test_build_tabular_dataset_keeps_x_y_meta_aligned() -> None:
    df = _make_panel_df()
    builder = _make_builder()

    dataset = builder.build_tabular_dataset(df)

    _assert_tabular_alignment(dataset)


def test_build_sequence_dataset_keeps_x_y_meta_aligned() -> None:
    df = _make_panel_df()
    seq_len = 3
    builder = _make_builder(seq_len=seq_len)

    dataset = builder.build_sequence_dataset(df)

    _assert_sequence_alignment(dataset, seq_len=seq_len)


def test_split_built_dataset_keeps_tabular_alignment() -> None:
    df = _make_panel_df()
    builder = _make_builder()

    dataset = builder.build_tabular_dataset(df)

    train_data, valid_data, test_data = builder.split_built_dataset(
        dataset=dataset,
        train_end="2020-01-03",
        valid_end="2020-01-05",
    )

    _assert_tabular_alignment(train_data)
    _assert_tabular_alignment(valid_data)
    _assert_tabular_alignment(test_data)

    train_dates = pd.to_datetime(train_data.meta["date"])
    valid_dates = pd.to_datetime(valid_data.meta["date"])
    test_dates = pd.to_datetime(test_data.meta["date"])

    assert train_dates.max() <= pd.Timestamp("2020-01-03")
    assert valid_dates.min() > pd.Timestamp("2020-01-03")
    assert valid_dates.max() <= pd.Timestamp("2020-01-05")
    assert test_dates.min() > pd.Timestamp("2020-01-05")


def test_split_built_dataset_keeps_sequence_alignment() -> None:
    df = _make_panel_df()
    seq_len = 3
    builder = _make_builder(seq_len=seq_len)

    dataset = builder.build_sequence_dataset(df)

    train_data, valid_data, test_data = builder.split_built_dataset(
        dataset=dataset,
        train_end="2020-01-04",
        valid_end="2020-01-06",
    )

    _assert_sequence_alignment(train_data, seq_len=seq_len)
    _assert_sequence_alignment(valid_data, seq_len=seq_len)
    _assert_sequence_alignment(test_data, seq_len=seq_len)

    train_dates = pd.to_datetime(train_data.meta["date"])
    valid_dates = pd.to_datetime(valid_data.meta["date"])
    test_dates = pd.to_datetime(test_data.meta["date"])

    assert train_dates.max() <= pd.Timestamp("2020-01-04")
    assert valid_dates.min() > pd.Timestamp("2020-01-04")
    assert valid_dates.max() <= pd.Timestamp("2020-01-06")
    assert test_dates.min() > pd.Timestamp("2020-01-06")


def test_subset_built_dataset_keeps_alignment() -> None:
    df = _make_panel_df()
    builder = _make_builder()

    dataset = builder.build_tabular_dataset(df)

    mask = dataset.meta["stock_id"].isin(["A", "C"]).to_numpy()
    subset = builder._subset_built_dataset(dataset, mask)

    _assert_tabular_alignment(subset)

    assert set(subset.meta["stock_id"]) == {"A", "C"}
    assert subset.X.shape[0] == int(mask.sum())


def test_subset_built_dataset_rejects_wrong_mask_length() -> None:
    df = _make_panel_df()
    builder = _make_builder()

    dataset = builder.build_tabular_dataset(df)
    bad_mask = np.array([True, False])

    with pytest.raises(ValueError, match="mask length mismatch"):
        builder._subset_built_dataset(dataset, bad_mask)


def test_subset_built_dataset_rejects_non_1d_mask() -> None:
    df = _make_panel_df()
    builder = _make_builder()

    dataset = builder.build_tabular_dataset(df)
    bad_mask = np.ones((dataset.X.shape[0], 1), dtype=bool)

    with pytest.raises(ValueError, match="mask must be 1D"):
        builder._subset_built_dataset(dataset, bad_mask)