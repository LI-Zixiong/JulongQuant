"""
Stock selection pipeline.

Reduces the full stock universe to a representative subset for efficient
model experimentation.

Stages:
  1. Load and preprocess raw panel data
  2. Compute per-stock metrics
  3. Apply hard filters: ST ratio, low price, insufficient history, data quality
  4. Compute composite score: data quality + liquidity + stability + size quality
  5. Select main pool with industry stratification
  6. Add a protected new-stock pool
  7. Fill any remaining slots by composite score
  8. Save selected panel, selected stock list, and distribution report

Outputs:
  dataset/input/selected_1500.parquet
  dataset/input/selected_1500_stock_ids.csv
  reports/selection_report.md
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_panel_data
from src.data.preprocess import PreprocessConfig, preprocess_panel_data


@dataclass
class SelectionConfig:
    data_path: str = "dataset/input/A_2010_2020.parquet"
    output_path: str = "dataset/input/selected_1500.parquet"
    selected_ids_path: str = "dataset/input/selected_1500_stock_ids.csv"
    report_path: str = "reports/selection_report.md"

    date_col: str = "time"
    stock_col: str = "stock_id"
    target_col: str = "1d_next_raw"
    industry_col: str = "industry_code"
    st_col: str = "st_status"

    target_stocks: int = 1500
    new_stock_pct: float = 0.05
    new_stock_min_days: int = 180
    new_stock_max_days: int = 252  # new-stock protection: 180-251 days

    main_min_days: int = 252
    min_valid_ratio: float = 0.75
    min_price_median: float = 2.0
    max_st_ratio: float = 0.10

    feature_cols: tuple[str, ...] = (
        "SIZE", "SIZENL", "LIQUIDITY", "BETA", "RESVOL",
        "MOMENTUM", "LEVERAGE", "VALUE", "EARNYLD", "GROWTH",
        "LTREV", "STREV",
    )


def _load_and_clean(config: SelectionConfig) -> pd.DataFrame:
    raw_df = load_panel_data(config.data_path)

    if config.date_col not in raw_df.columns and config.date_col in raw_df.index.names:
        raw_df = raw_df.reset_index()

    preprocess_config = PreprocessConfig(
        date_col=config.date_col,
        stock_col=config.stock_col,
        target_col=config.target_col,
        feature_cols=list(config.feature_cols),
        meta_cols=[config.industry_col, "price", config.st_col],
        replace_inf_with_nan=True,
        drop_rows_with_missing_keys=True,
        drop_rows_with_missing_features=False,
        drop_rows_with_missing_target=False,
        duplicate_policy="raise",
        sort_values=True,
    )

    result = preprocess_panel_data(raw_df, preprocess_config)
    return result.df


def _compute_st_ratio(s: pd.Series) -> float:
    """Return the fraction of rows that are marked as ST-like status."""
    values = s.fillna("").astype(str).str.strip().str.upper()
    st_like = values.isin({"ST", "*ST", "SST", "1", "TRUE", "T", "YES", "Y"})
    return float(st_like.mean())


def _compute_stock_metrics(df: pd.DataFrame, config: SelectionConfig) -> pd.DataFrame:
    date_col = config.date_col
    stock_col = config.stock_col
    target_col = config.target_col

    required_for_validity = [target_col, *config.feature_cols]
    valid_cols = [col for col in required_for_validity if col in df.columns]
    missing_valid_cols = sorted(set(required_for_validity) - set(valid_cols))
    if missing_valid_cols:
        raise ValueError(f"Missing required validity columns: {missing_valid_cols}")

    total_dates = df[date_col].nunique()
    grouped = df.groupby(stock_col, sort=False)

    n_dates_appeared = grouped[date_col].nunique()

    # A valid modeling row must have both target and all factor values available.
    # This makes valid_ratio a true target + factor completeness ratio, instead
    # of only checking the target column.
    row_valid = df[valid_cols].notna().all(axis=1)
    n_valid_days = row_valid.groupby(df[stock_col]).sum().astype(int)
    valid_ratio = n_valid_days / n_dates_appeared.replace(0, np.nan)

    n_valid_target = grouped[target_col].apply(lambda s: int(s.notna().sum()))
    st_fraction = grouped[config.st_col].apply(_compute_st_ratio)
    price_median = grouped["price"].median()

    size_median = grouped["SIZE"].median()
    liquidity_median = grouped["LIQUIDITY"].median()
    resvol_median = grouped["RESVOL"].median()

    last_industry = grouped[config.industry_col].last()

    metrics = pd.DataFrame(
        {
            "n_dates_appeared": n_dates_appeared,
            "n_valid_target": n_valid_target,
            "n_valid_days": n_valid_days,
            "valid_ratio": valid_ratio,
            "st_ratio": st_fraction,
            "price_median": price_median,
            "size_median": size_median,
            "liquidity_median": liquidity_median,
            "resvol_median": resvol_median,
            "industry_code": last_industry,
        }
    ).reset_index()

    metrics["total_dates"] = total_dates
    return metrics


def _apply_hard_filters(
    metrics: pd.DataFrame, config: SelectionConfig
) -> tuple[pd.DataFrame, dict[str, int]]:
    stats: dict[str, int] = {"total_before": len(metrics)}
    current = metrics.copy()

    mask = current["st_ratio"] > config.max_st_ratio
    stats["st_removed"] = int(mask.sum())
    current = current[~mask].copy()
    stats["after_st"] = len(current)

    mask = current["price_median"].isna() | (current["price_median"] < config.min_price_median)
    stats["price_removed"] = int(mask.sum())
    current = current[~mask].copy()
    stats["after_price"] = len(current)

    mask = current["n_valid_days"] < config.new_stock_min_days
    stats["valid_days_removed"] = int(mask.sum())
    current = current[~mask].copy()
    stats["after_valid_days"] = len(current)

    mask = current["valid_ratio"].isna() | (current["valid_ratio"] < config.min_valid_ratio)
    stats["valid_ratio_removed"] = int(mask.sum())
    current = current[~mask].copy()
    stats["passed_hard_filters"] = len(current)

    # Keep a fresh RangeIndex and make every later selection/report use this
    # filtered DataFrame, not the original metrics DataFrame.
    return current.reset_index(drop=True), stats


def _pct_rank(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    pct = s.rank(pct=True)
    if not higher_is_better:
        pct = 1.0 - pct
    return pct.fillna(0.0)


def _compute_composite_score(metrics: pd.DataFrame) -> pd.Series:
    # Liquidity: assumes higher percentile = more liquid.
    # If the LIQUIDITY factor is inverted in your dataset, change
    # higher_is_better=True to higher_is_better=False here.
    liquidity_pct = _pct_rank(metrics["liquidity_median"], higher_is_better=True)

    # Lower RESVOL is preferred for stability.
    stability = _pct_rank(metrics["resvol_median"], higher_is_better=False)

    valid_ratio_pct = _pct_rank(metrics["valid_ratio"], higher_is_better=True)
    n_valid_pct = _pct_rank(metrics["n_valid_days"], higher_is_better=True)
    size_pct = _pct_rank(metrics["size_median"], higher_is_better=True)

    composite = (
        valid_ratio_pct * 0.25
        + n_valid_pct * 0.25
        + liquidity_pct * 0.30
        + stability * 0.10
        + size_pct * 0.10
    )

    return composite.astype(float)


def _classify_pools(
    metrics: pd.DataFrame, config: SelectionConfig
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    composite = _compute_composite_score(metrics)

    is_main = metrics["n_valid_days"] >= config.main_min_days

    is_new = (
        (metrics["n_valid_days"] >= config.new_stock_min_days)
        & (metrics["n_valid_days"] < config.new_stock_max_days)
        & (metrics["valid_ratio"] >= config.min_valid_ratio)
    )

    is_excluded = ~is_main & ~is_new

    return composite, is_main, is_new, is_excluded


def _industry_stratified_select(
    metrics: pd.DataFrame,
    composite: pd.Series,
    candidates: pd.Series,
    n_slots: int,
) -> pd.Index:
    eligible = metrics.loc[candidates].copy()
    if eligible.empty or n_slots <= 0:
        return pd.Index([], dtype=int)

    eligible["composite"] = composite.loc[eligible.index]
    eligible["industry_code"] = eligible["industry_code"].fillna("missing_industry")

    industry_counts = eligible["industry_code"].value_counts()
    industry_weights = industry_counts / industry_counts.sum()
    allocation = (industry_weights * n_slots).round().astype(int)

    # Adjust rounding so sum equals n_slots.
    diff = n_slots - int(allocation.sum())
    if diff != 0:
        order = allocation.sort_values(ascending=(diff < 0)).index.tolist()
        step = 1 if diff > 0 else -1
        remaining = abs(diff)
        for industry in order:
            if remaining == 0:
                break
            if allocation[industry] + step >= 0:
                allocation[industry] += step
                remaining -= 1

    selected: list[int] = []
    for industry, quota in allocation.items():
        if quota <= 0:
            continue
        pool = eligible[eligible["industry_code"] == industry]
        top = pool.nlargest(min(quota, len(pool)), "composite")
        selected.extend(top.index.tolist())

    return pd.Index(selected, dtype=int)


def _safe_numeric_summary(s: pd.Series, fn) -> float:
    clean = s.dropna()
    if clean.empty:
        return float("nan")
    return float(fn(clean))


def _build_distribution_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
    hard_filter_stats: dict[str, int],
    config: SelectionConfig,
    main_count: int,
    new_count: int,
    fallback_count: int,
) -> str:
    ind_before = before["industry_code"].value_counts()
    ind_after = after["industry_code"].value_counts()

    lines = [
        "# Stock Selection Report",
        "",
        "## Hard Filter Summary",
        "",
        "| Stage | Stocks Removed | Stocks Remaining |",
        "|---|---:|---:|",
        f"| Before filters | — | {hard_filter_stats['total_before']} |",
        f"| ST ratio > {config.max_st_ratio} | {hard_filter_stats['st_removed']} | {hard_filter_stats['after_st']} |",
        f"| price_median < {config.min_price_median} or missing | {hard_filter_stats['price_removed']} | {hard_filter_stats['after_price']} |",
        f"| n_valid_days < {config.new_stock_min_days} | {hard_filter_stats['valid_days_removed']} | {hard_filter_stats['after_valid_days']} |",
        f"| valid_ratio < {config.min_valid_ratio} or missing | {hard_filter_stats['valid_ratio_removed']} | {hard_filter_stats['passed_hard_filters']} |",
        "",
        "## Selection Summary",
        "",
        f"- Hard-filter survivors: {hard_filter_stats['passed_hard_filters']}",
        f"- Main industry-stratified pool: {main_count}",
        f"- New-stock protected pool: {new_count}",
        f"- Fallback filled pool: {fallback_count}",
        f"- Selected: {len(after)} stocks ({config.target_stocks} target)",
        "",
        "## Industry Distribution",
        "",
        "| Industry | Before | After |",
        "|---|---:|---:|",
    ]

    for industry in ind_before.index.union(ind_after.index).sort_values():
        b = int(ind_before.get(industry, 0))
        a = int(ind_after.get(industry, 0))
        lines.append(f"| {industry} | {b} | {a} |")

    for col, label in [
        ("size_median", "SIZE"),
        ("liquidity_median", "LIQUIDITY"),
        ("resvol_median", "RESVOL"),
        ("n_valid_days", "n_valid_days"),
        ("valid_ratio", "valid_ratio"),
    ]:
        lines.append("")
        lines.append(f"## {label} Distribution")
        lines.append("")
        lines.append("| Stat | Before | After |")
        lines.append("|---|---:|---:|")
        for stat_name, fn in [
            ("mean", np.mean),
            ("std", np.std),
            ("min", np.min),
            ("25%", lambda x: np.percentile(x, 25)),
            ("50%", lambda x: np.percentile(x, 50)),
            ("75%", lambda x: np.percentile(x, 75)),
            ("max", np.max),
        ]:
            b_val = _safe_numeric_summary(before[col], fn)
            a_val = _safe_numeric_summary(after[col], fn)
            lines.append(f"| {stat_name} | {b_val:.4f} | {a_val:.4f} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- valid_ratio is computed from rows where the target and all configured "
        "factor columns are non-missing."
    )
    lines.append(
        "- LIQUIDITY direction assumes higher percentile = more liquid. If the "
        "factor is inverted, change the percentile direction in _compute_composite_score."
    )
    lines.append(
        "- SIZE score (size_quality) rewards larger market cap as a proxy for "
        "tradability, not as a diversity measure."
    )
    lines.append(
        f"- New-stock protection reserves ~{config.new_stock_pct:.0%} of slots "
        f"for stocks with {config.new_stock_min_days}–{config.new_stock_max_days - 1} "
        f"valid days and valid_ratio >= {config.min_valid_ratio}."
    )

    return "\n".join(lines)


def main() -> None:
    config = SelectionConfig()

    print("Stage 1: Loading and cleaning data...")
    df = _load_and_clean(config)
    print(f"  Loaded {len(df):,} rows")

    print("Stage 2: Computing per-stock metrics...")
    metrics = _compute_stock_metrics(df, config)
    n_before_select = len(metrics)
    print(f"  {len(metrics)} unique stocks")

    print("Stage 3: Applying hard filters...")
    filtered, hard_filter_stats = _apply_hard_filters(metrics, config)
    print(f"  {len(filtered)} stocks passed hard filters")
    print(f"    Removed {hard_filter_stats['st_removed']:>5}  (ST ratio > {config.max_st_ratio})")
    print(f"    Removed {hard_filter_stats['price_removed']:>5}  (price_median < {config.min_price_median} or missing)")
    print(f"    Removed {hard_filter_stats['valid_days_removed']:>5}  (n_valid_days < {config.new_stock_min_days})")
    print(f"    Removed {hard_filter_stats['valid_ratio_removed']:>5}  (valid_ratio < {config.min_valid_ratio} or missing)")

    print("Stage 4: Computing composite scores...")
    composite, is_main, is_new, is_excluded = _classify_pools(filtered, config)

    n_main_candidates = int(is_main.sum())
    n_new_candidates = int(is_new.sum())
    print(f"  Main pool candidates: {n_main_candidates}")
    print(f"  New-stock pool candidates: {n_new_candidates}")
    print(f"  Excluded after hard filters: {int(is_excluded.sum())}")

    n_new_slots = int(round(config.target_stocks * config.new_stock_pct))
    n_main_slots = config.target_stocks - n_new_slots
    print(f"  Allocating {n_main_slots} main + {n_new_slots} new-stock = {config.target_stocks}")

    print("Stage 5: Industry-stratified selection (main pool)...")
    selected_main = _industry_stratified_select(
        filtered, composite, is_main, n_main_slots
    )

    print("Stage 6: New-stock protection selection...")
    selected_set = set(selected_main.tolist())
    new_candidates = filtered.loc[is_new & ~filtered.index.isin(selected_set)].copy()
    new_candidates["composite"] = composite.loc[new_candidates.index]
    selected_new = new_candidates.nlargest(
        min(n_new_slots, len(new_candidates)), "composite"
    ).index

    selected_set.update(selected_new.tolist())
    final_indices = pd.Index(list(selected_set), dtype=int)

    print("Stage 7: Fallback fill, if needed...")
    fallback_indices = pd.Index([], dtype=int)
    if len(final_indices) < config.target_stocks:
        missing = config.target_stocks - len(final_indices)
        fallback_candidates = filtered.loc[~filtered.index.isin(final_indices)].copy()
        fallback_candidates["composite"] = composite.loc[fallback_candidates.index]
        fallback_indices = fallback_candidates.nlargest(missing, "composite").index
        final_indices = pd.Index(final_indices.tolist() + fallback_indices.tolist(), dtype=int)

    if len(final_indices) > config.target_stocks:
        final_with_score = filtered.loc[final_indices].copy()
        final_with_score["composite"] = composite.loc[final_indices]
        final_indices = final_with_score.nlargest(config.target_stocks, "composite").index

    selection_group = pd.Series(index=filtered.index, dtype="object")
    selection_group.loc[selected_main] = "main_industry_stratified"
    selection_group.loc[selected_new] = "new_stock_protected"
    selection_group.loc[fallback_indices] = "fallback_filled"

    # Critical fix: final_indices is based on `filtered`, so every output and
    # report must use `filtered.loc[final_indices]`, not `metrics.loc[final_indices]`.
    filtered_selected = filtered.loc[final_indices].copy()
    filtered_selected["composite"] = composite.loc[final_indices]
    filtered_selected["selection_group"] = selection_group.loc[final_indices].values

    if filtered_selected["n_valid_days"].min() < config.new_stock_min_days:
        raise RuntimeError(
            "Selection sanity check failed: selected stocks include n_valid_days "
            f"below {config.new_stock_min_days}."
        )
    if len(filtered_selected) != config.target_stocks:
        raise RuntimeError(
            f"Selection sanity check failed: selected {len(filtered_selected)} stocks, "
            f"target is {config.target_stocks}."
        )

    main_count = int((filtered_selected["selection_group"] == "main_industry_stratified").sum())
    new_count = int((filtered_selected["selection_group"] == "new_stock_protected").sum())
    fallback_count = int((filtered_selected["selection_group"] == "fallback_filled").sum())
    print(f"  Main: {main_count}, New: {new_count}, Fallback: {fallback_count}, Total: {len(filtered_selected)}")

    print("Stage 8: Building output parquet...")
    selected_stock_ids = set(filtered_selected[config.stock_col])
    output_df = df[df[config.stock_col].isin(selected_stock_ids)].copy()
    output_df = output_df.sort_values(
        [config.date_col, config.stock_col], kind="mergesort"
    ).reset_index(drop=True)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(output_path, index=False)
    print(f"  Saved {len(output_df):,} rows to {output_path}")

    selected_ids_path = Path(config.selected_ids_path)
    selected_ids_path.parent.mkdir(parents=True, exist_ok=True)
    id_cols = [
        config.stock_col,
        "industry_code",
        "composite",
        "valid_ratio",
        "n_valid_days",
        "n_dates_appeared",
        "n_valid_target",
        "price_median",
        "st_ratio",
        "liquidity_median",
        "resvol_median",
        "size_median",
        "selection_group",
    ]
    filtered_selected[id_cols].sort_values(
        ["selection_group", "composite"], ascending=[True, False]
    ).to_csv(selected_ids_path, index=False)
    print(f"  Saved selected stock ids to {selected_ids_path}")

    print("Stage 9: Generating distribution report...")
    before_metrics = metrics.drop(columns=["total_dates"])
    after_metrics = filtered_selected.drop(columns=["total_dates"])
    report = _build_distribution_report(
        before_metrics,
        after_metrics,
        hard_filter_stats,
        config,
        main_count=main_count,
        new_count=new_count,
        fallback_count=fallback_count,
    )

    report_path = Path(config.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report saved to {report_path}")

    print(f"\nDone: {n_before_select} → {len(filtered_selected)} stocks selected.")
    print(f"Data path for ExperimentConfig: {config.output_path}")


if __name__ == "__main__":
    main()
