"""
Build unified daily factor panel from raw CSMAR data files.

Input:  24 CSV files in dataset/input/original_data/
Output:
  dataset/processed/csmar_daily_raw_panel.parquet    (2012+, warm-up)
  dataset/processed/csmar_factor_ready_panel.parquet  (2018+, model input)

Pipeline (V0):
  0. Stock master + calendar   (TRD_Co, TRD_Cale)
  1. Daily price/volume/market (TRD_Dalyr x7, TRD_AdjustFactor, TRD_Nrrate, TRD_Cndalym)
  2. Quarterly financials      (FS_Combas, FS_Comins, FS_Comscfi — left-merge)
  3. Sanity checks

Governance / equity change / insider are deferred to V1 (flags default False).
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.seed import set_seed

DATA_DIR = Path("dataset/input/original_data")
OUTPUT_DIR = Path("dataset/processed")


@dataclass
class FactorPanelConfig:
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR

    # Date ranges
    start_date: str = "2012-01-01"       # earliest raw data loaded
    model_start_date: str = "2018-01-01"  # factor-ready panel start
    end_date: str = "2026-05-18"

    date_col: str = "time"
    stock_col: str = "stock_id"

    # Stock universe
    exclude_financials: bool = True
    min_listing_days: int = 180

    # V0: defer governance / equity / insider
    include_governance: bool = False
    include_equity_change: bool = False
    include_insider: bool = False

    seed: int = 42


# ========  Step 0: stock master + calendar  ========

def _build_stock_master(config: FactorPanelConfig) -> pd.DataFrame:
    """Load TRD_Co as reference; do NOT filter by Statco historically."""
    df = pd.read_csv(config.data_dir / "TRD_Co.csv")

    df["Stkcd"] = df["Stkcd"].astype(str).str.strip()
    df["Listdt"] = pd.to_datetime(df["Listdt"], errors="coerce")

    is_financial_mask = df["Nnindcd"].astype(str).str.startswith("J", na=False)

    df = df.rename(columns={
        "Stkcd": config.stock_col,
        "Nnindnme": "industry_csrc_2012",
        "Listdt": "list_date",
        "Stknme": "stock_name_short",
        "Statco": "statco_current",
    })

    keep_cols = [config.stock_col, "stock_name_short", "industry_csrc_2012",
                 "list_date", "Markettype", "statco_current"]
    df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)
    df["is_financial"] = is_financial_mask.values
    return df


def _build_calendar(config: FactorPanelConfig) -> pd.DatetimeIndex:
    df = pd.read_csv(config.data_dir / "TRD_Cale.csv")
    df = df[df["State"] == "O"].copy()
    dates = pd.to_datetime(df["Clddt"], errors="coerce").dropna()
    mask = (dates >= config.start_date) & (dates <= config.end_date)
    return pd.DatetimeIndex(dates[mask].sort_values().unique(), name=config.date_col)


# ========  Step 1: daily price/volume/market  ========

def _load_dalyr(config: FactorPanelConfig) -> pd.DataFrame:
    # Scan the root and the three date-split subdirectories
    search_dirs = [
        config.data_dir,
        config.data_dir / "Dalyr1",
        config.data_dir / "Dalyr2",
        config.data_dir / "Dalyr3",
    ]
    files = []
    for d in search_dirs:
        if d.exists():
            files.extend(d.glob("TRD_Dalyr*.csv"))
    files = sorted(set(files))
    print(f"  Found {len(files)} TRD_Dalyr files")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df = df[df["Markettype"].isin({1, 4, 16, 32})].copy()
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(df):,} rows (A-share only)")
    df["Stkcd"] = df["Stkcd"].astype(str).str.strip()
    df["Trddt"] = pd.to_datetime(df["Trddt"])
    return df.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)


def _load_adjust_factor(config: FactorPanelConfig) -> pd.DataFrame:
    df = pd.read_csv(config.data_dir / "TRD_AdjustFactor.csv")
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["TradingDate"] = pd.to_datetime(df["TradingDate"])
    return df[["Symbol", "TradingDate", "CumulateFwardFactor"]].rename(
        columns={"Symbol": "Stkcd", "TradingDate": "Trddt"}
    )


def _load_nrrate(config: FactorPanelConfig) -> pd.DataFrame:
    df = pd.read_csv(config.data_dir / "TRD_Nrrate.csv")
    df["Clsdt"] = pd.to_datetime(df["Clsdt"])
    return df[["Clsdt", "Nrrdaydt"]].rename(columns={"Clsdt": "Trddt"})


def _load_market_return(config: FactorPanelConfig) -> pd.DataFrame:
    """
    TRD_Cndalym: combined market daily stats.
    Cdretwdtl = 考虑现金红利再投资的综合日市场回报率(总市值加权平均法)
    Markettype=69 = 沪深京A股市场
    """
    df = pd.read_csv(config.data_dir / "TRD_Cndalym.csv")
    # Markettype=53: 沪深A股和创业板和科创板 (matches our {1,4,16,32} filter)
    df = df[df["Markettype"] == 53].copy()
    df["Trddt"] = pd.to_datetime(df["Trddt"])
    return df[["Trddt", "Cdretwdtl"]].rename(
        columns={"Trddt": config.date_col, "Cdretwdtl": "mkt_ret_vw"}
    )


def _build_daily_layer(config: FactorPanelConfig) -> pd.DataFrame:
    print("  Loading TRD_Dalyr...")
    daily = _load_dalyr(config)

    print("  Merging adjust factor...")
    adj = _load_adjust_factor(config)
    adj = adj.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)
    daily = daily.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)
    daily = pd.merge_asof(
        daily.sort_values("Trddt"),
        adj.sort_values("Trddt"),
        by="Stkcd", on="Trddt", direction="backward",
    )

    print("  Merging risk-free rate...")
    nrrate = _load_nrrate(config)
    daily = daily.merge(nrrate, on="Trddt", how="left")

    daily = daily.rename(columns={
        "Trddt": config.date_col,
        "Stkcd": config.stock_col,
        "Opnprc": "open",
        "Hiprc": "high",
        "Loprc": "low",
        "Clsprc": "close",
        "Dnshrtrd": "volume",
        "Dnvaltrd": "amount",
        "Dsmvosd": "mktcap_float",       # unit: thousand yuan → *1000 below
        "Dretwd": "ret_daily",
        "Adjprcwd": "close_adj",
        "Trdsta": "trade_status",
        "ChangeRatio": "change_ratio",
        "LimitStatus": "limit_status",
        "Capchgdt": "cap_chg_date",
        "CumulateFwardFactor": "cum_adj_factor",
        "Nrrdaydt": "rf_daily",
    })

    daily["mktcap_float"] = daily["mktcap_float"] * 1000.0  # thousand → yuan

    daily[config.date_col] = pd.to_datetime(daily[config.date_col])
    daily["cap_chg_date"] = pd.to_datetime(daily.get("cap_chg_date"), errors="coerce")

    print("  Merging market return (TRD_Cndalym)...")
    mkt = _load_market_return(config)
    daily = daily.merge(mkt, on=config.date_col, how="left")

    daily = daily[
        (daily[config.date_col] >= config.start_date)
        & (daily[config.date_col] <= config.end_date)
    ]

    daily = daily.drop_duplicates(subset=[config.date_col, config.stock_col])
    daily = daily.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)

    keep_cols = [
        config.date_col, config.stock_col,
        "open", "high", "low", "close", "volume", "amount",
        "mktcap_float", "ret_daily", "close_adj", "change_ratio", "limit_status",
        "trade_status", "cap_chg_date", "cum_adj_factor", "rf_daily",
        "mkt_ret_vw", "Markettype",
    ]
    available = [c for c in keep_cols if c in daily.columns]
    return daily[available]


# ========  Step 2: quarterly financials  ========

def _build_mktcap_total(
    daily: pd.DataFrame, config: FactorPanelConfig
) -> pd.DataFrame:
    """Merge total shares from TRD_Capchg and compute mktcap_total."""
    cap = pd.read_csv(config.data_dir / "TRD_Capchg.csv")
    cap["Stkcd"] = cap["Stkcd"].astype(str).str.strip()
    cap["Shrchgdt"] = pd.to_datetime(cap["Shrchgdt"], errors="coerce")

    # Keep only the latest total shares per stock per date
    cap = cap.sort_values(["Stkcd", "Shrchgdt"]).reset_index(drop=True)
    cap = cap.rename(columns={
        "Stkcd": config.stock_col,
        "Shrchgdt": "chg_date",
        "Nshrttl": "total_shares",
    })
    cap = cap[[config.stock_col, "chg_date", "total_shares"]].dropna(
        subset=["total_shares"]
    )

    # Ffill total_shares to each trading day via merge_asof
    daily = daily.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)
    cap = cap.sort_values([config.stock_col, "chg_date"]).reset_index(drop=True)

    daily = pd.merge_asof(
        daily.sort_values(config.date_col),
        cap.sort_values("chg_date"),
        by=config.stock_col,
        left_on=config.date_col,
        right_on="chg_date",
        direction="backward",
    )
    daily = daily.drop(columns=["chg_date"], errors="ignore")

    daily["mktcap_total"] = daily["total_shares"] * daily["close"]
    daily["mktcap_total"] = daily["mktcap_total"].fillna(daily["mktcap_float"])
    return daily


def _shift_to_available_date(accper: pd.Series) -> pd.Series:
    """Map accounting period end → earliest conservative available date."""
    m = accper.dt.month
    y = accper.dt.year

    result = accper.copy()
    result[m == 3] = pd.to_datetime(y[m == 3].astype(str) + "-05-01")
    result[m == 6] = pd.to_datetime(y[m == 6].astype(str) + "-09-01")
    result[m == 9] = pd.to_datetime(y[m == 9].astype(str) + "-11-01")
    result[m == 12] = pd.to_datetime((y[m == 12] + 1).astype(str) + "-05-01")

    non_std = ~m.isin({3, 6, 9, 12})
    if non_std.any():
        result[non_std] = accper[non_std] + pd.DateOffset(months=4)
        print(f"    [WARNING] Non-standard Accper months: {accper[non_std].dt.month.unique().tolist()} "
              f"(n={non_std.sum()}), using Accper+4mo fallback")

    return result


def _load_financial(config: FactorPanelConfig) -> pd.DataFrame:
    bas = pd.read_csv(config.data_dir / "FS_Combas.csv")
    ins = pd.read_csv(config.data_dir / "FS_Comins.csv")
    scf = pd.read_csv(config.data_dir / "FS_Comscfi.csv")
    scf_direct = pd.read_csv(config.data_dir / "FS_Comscfd.csv")

    bas = bas[bas["Typrep"] == "A"].copy()
    ins = ins[ins["Typrep"] == "A"].copy()
    scf = scf[scf["Typrep"] == "A"].copy()
    scf_direct = scf_direct[scf_direct["Typrep"] == "A"].copy()

    for d in (bas, ins, scf, scf_direct):
        d["Stkcd"] = d["Stkcd"].astype(str).str.strip()
        d["Accper"] = pd.to_datetime(d["Accper"])

    # Verify Typrep filter worked
    for name, d in [("Combas", bas), ("Comins", ins), ("Comscfi", scf)]:
        dup = d.groupby(["Stkcd", "Accper"]).size()
        if (dup > 1).any():
            print(f"    [WARNING] {name} has {int((dup>1).sum())} duplicate Stkcd-Accper after Typrep filter")

    bas = bas.rename(columns={
        "A001000000": "total_assets",
        "A002000000": "total_liabilities",
        "A003000000": "total_equity",
        "A003100000": "equity_parent",
        "A001100000": "current_assets",
        "A001200000": "noncurrent_assets",
        "A002100000": "current_liabilities",
        "A002200000": "noncurrent_liabilities",
        "A001101000": "cash",
        "A001111000": "receivables",
        "A001123000": "inventory",
        "A002107000": "payables_trade",
    })

    ins = ins.rename(columns={
        "B001100000": "revenue_total",
        "B001101000": "revenue",
        "B001201000": "cost_revenue",
        "B001209000": "sell_expense",
        "B001210000": "admin_expense",
        "B001216000": "rd_expense",
        "B001211000": "finance_expense",
        "B001300000": "operating_profit",
        "B002000000": "net_profit",
        "B002000101": "net_profit_parent",
        "B002100000": "income_tax",
        "B003000000": "eps_basic",
    })

    scf = scf.rename(columns={
        "D000101000": "cf_ni",
        "D000102000": "cf_impairment",
        "D000103000": "cf_depreciation",
        "D000104000": "cf_amort_intangible",
        "D000105000": "cf_amort_lte",
        "D000113000": "cf_inventory_change",
        "D000114000": "cf_receivables_change",
        "D000115000": "cf_payables_change",
        "D000100000": "cf_operating",
        "D000200000": "cf_net_change",
    })

    bas_cols = [c for c in [
        "Stkcd", "Accper", "total_assets", "total_liabilities", "total_equity",
        "equity_parent", "current_assets", "noncurrent_assets",
        "current_liabilities", "noncurrent_liabilities",
        "cash", "receivables", "inventory", "payables_trade",
    ] if c in bas.columns]
    fin = bas[bas_cols]

    ins_cols = [c for c in [
        "Stkcd", "Accper", "revenue_total", "revenue", "cost_revenue",
        "sell_expense", "admin_expense", "rd_expense", "finance_expense",
        "operating_profit", "net_profit", "net_profit_parent",
        "income_tax", "eps_basic",
    ] if c in ins.columns]
    fin = fin.merge(ins[ins_cols], on=["Stkcd", "Accper"], how="inner")
    print(f"    bas+ins merge: {len(fin)} rows")

    # scf: left merge to avoid dropping rows without cashflow data
    scf_cols = [c for c in [
        "Stkcd", "Accper", "cf_ni", "cf_impairment", "cf_depreciation",
        "cf_amort_intangible", "cf_amort_lte",
        "cf_inventory_change", "cf_receivables_change", "cf_payables_change",
        "cf_operating", "cf_net_change",
    ] if c in scf.columns]
    fin = fin.merge(scf[scf_cols], on=["Stkcd", "Accper"], how="left")
    print(f"    +scf (indirect) left merge: {len(fin)} rows")

    # Direct method cash flow: rename key columns, merge, then coalesce
    scf_direct = scf_direct.rename(columns={
        "C001000000": "cf_operating_direct",
        "C001001000": "cf_sales_cash",
        "C001012000": "cf_tax_rebate",
        "C001014000": "cf_purchase_cash",
        "C001020000": "cf_wages_cash",
        "C001021000": "cf_taxes_cash",
        "C002001000": "cf_invest_recover",
        "C002002000": "cf_invest_income",
        "C002006000": "cf_capex",
        "C002007000": "cf_invest_paid",
        "C003002000": "cf_borrow",
        "C003005000": "cf_repay_debt",
        "C003006000": "cf_dividend_paid",
        "C003008000": "cf_equity_issue",
        "C005000000": "cf_net_change_direct",
    })
    scfd_cols = [c for c in [
        "Stkcd", "Accper", "cf_operating_direct", "cf_sales_cash",
        "cf_tax_rebate", "cf_purchase_cash", "cf_wages_cash", "cf_taxes_cash",
        "cf_invest_recover", "cf_invest_income", "cf_capex", "cf_invest_paid",
        "cf_borrow", "cf_repay_debt", "cf_dividend_paid", "cf_equity_issue",
        "cf_net_change_direct",
    ] if c in scf_direct.columns]
    fin = fin.merge(scf_direct[scfd_cols], on=["Stkcd", "Accper"], how="left")
    print(f"    +scfd (direct) left merge: {len(fin)} rows")

    # Fill indirect cf_operating gaps with direct method values
    n_before = fin["cf_operating"].notna().sum()
    fin["cf_operating"] = fin["cf_operating"].fillna(fin["cf_operating_direct"])
    n_after = fin["cf_operating"].notna().sum()
    print(f"    cf_operating coverage: {n_before} → {n_after} (+{n_after - n_before} from direct method)")

    fin = fin.rename(columns={"Stkcd": config.stock_col, "Accper": "accper"})

    fin["accper"] = pd.to_datetime(fin["accper"])
    fin["available_date"] = _shift_to_available_date(fin["accper"])

    # TODO: cumulative → single-quarter → TTM before computing EARNYLD/GROWTH
    return fin


def _ffill_financial_to_daily(
    daily: pd.DataFrame,
    financial: pd.DataFrame,
    config: FactorPanelConfig,
) -> pd.DataFrame:
    """Forward-fill quarterly financials onto daily panel."""
    fin_cols = [c for c in financial.columns if c not in (
        config.stock_col, "accper", "available_date"
    )]

    daily = daily.sort_values([config.stock_col, config.date_col]).reset_index(drop=True)
    financial = financial.sort_values([config.stock_col, "available_date"]).reset_index(drop=True)
    fin_dict = {s: g for s, g in financial.groupby(config.stock_col)}

    result_parts = []
    for stock, group in daily.groupby(config.stock_col):
        if stock not in fin_dict:
            group = group.copy()
            for c in fin_cols:
                group[c] = np.nan
            result_parts.append(group)
            continue

        stock_fin = fin_dict[stock].copy()
        stock_dates = group[config.date_col].values

        for c in fin_cols:
            fin_dates = stock_fin["available_date"].values
            fin_vals = stock_fin[c].values
            daily_vals = np.full(len(stock_dates), np.nan)
            if len(fin_vals) > 0:
                idx = np.searchsorted(fin_dates, stock_dates, side="right") - 1
                valid = idx >= 0
                daily_vals[valid] = fin_vals[idx[valid]]
            group = group.copy()
            group[c] = daily_vals

        result_parts.append(group)

    return pd.concat(result_parts, ignore_index=True)


# ========  Sanity checks  ========

def _run_sanity_checks(
    daily: pd.DataFrame,
    financial: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    config: FactorPanelConfig,
) -> None:
    print("\n========== Sanity Checks ==========")

    n = len(daily)
    n_stocks = daily[config.stock_col].nunique()
    dates = pd.DatetimeIndex(daily[config.date_col].unique()).sort_values()

    print(f"Rows: {n:,}")
    print(f"Stocks: {n_stocks}")
    print(f"Date range: {dates.min().date()} ~ {dates.max().date()}")
    print(f"Trading days in panel: {len(dates)}")
    print(f"Calendar trading days: {len(calendar)} (expected {len(dates)})")

    # Per-date stock count
    counts = daily.groupby(config.date_col).size()
    print(f"Per-date stock count — min={counts.min()} median={counts.median():.0f} mean={counts.mean():.0f} max={counts.max()}")

    # Missing rates
    check_cols = [
        "close", "close_adj", "ret_daily", "volume", "amount",
        "mktcap_float", "mktcap_total", "change_ratio", "limit_status",
        "trade_status", "rf_daily", "mkt_ret_vw",
        "total_assets", "total_liabilities", "equity_parent",
        "revenue_total", "net_profit_parent", "cf_operating",
    ]
    print("\nMissing rates:")
    for c in check_cols:
        if c not in daily.columns:
            print(f"  {c:25s}  NOT IN PANEL")
            continue
        rate = daily[c].isna().mean()
        print(f"  {c:25s}  {rate:.4f}")

    # Dedup check
    dup = daily.duplicated(subset=[config.date_col, config.stock_col]).sum()
    print(f"\nDuplicate date-stock rows: {dup}")

    # Financial coverage by year
    daily["_year"] = daily[config.date_col].dt.year
    fin_cols_in_panel = [c for c in check_cols if c in daily.columns and c.startswith(("total_", "equity", "revenue", "net_profit", "cf_"))]
    if fin_cols_in_panel:
        print("\nFinancial coverage by year:")
        for y in sorted(daily["_year"].dropna().unique()):
            yr = daily[daily["_year"] == y]
            cov = yr[fin_cols_in_panel[0]].notna().mean()
            print(f"  {int(y)}: {cov:.4f}")

    # available_date > accper sanity (from financial df itself)
    if not financial.empty:
        bad = financial["available_date"] <= financial["accper"]
        print(f"\nFinancial available_date <= accper: {bad.sum()} rows")
        if bad.any():
            print("  Sample:")
            print(financial.loc[bad, ["accper", "available_date"]].head())

    # mktcap_float sanity (quarterlies)
    if "mktcap_float" in daily.columns:
        mc = daily["mktcap_float"].dropna()
        qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        print("\nmktcap_float quantiles (yuan):")
        for q in qs:
            print(f"  {q:.0%}: {mc.quantile(q):,.0f}")

    daily.drop(columns=["_year"], inplace=True, errors="ignore")


def _sample_stock_trajectory(
    daily: pd.DataFrame, config: FactorPanelConfig, stock_id: str
) -> None:
    """Print a single stock's financial ffill trajectory for manual check."""
    s = daily[daily[config.stock_col] == stock_id].sort_values(config.date_col)
    if s.empty:
        print(f"  Stock {stock_id} not found")
        return
    mid = len(s) // 2
    fin_cols = [c for c in ["total_assets", "equity_parent", "revenue_total",
                              "net_profit_parent", "cf_operating"] if c in s.columns]
    sample = s[[config.date_col] + fin_cols].iloc[mid:mid + 10]
    print(f"\n  Stock {stock_id} trajectory sample:")
    print(sample.to_string(index=False))


# ========  Main  ========

def main() -> None:
    config = FactorPanelConfig()
    set_seed(config.seed)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 0 ----
    print("Step 0: Building stock master and calendar...")
    stock_master = _build_stock_master(config)
    calendar = _build_calendar(config)
    print(f"  {len(stock_master)} reference stocks ({stock_master['is_financial'].sum()} financial)")
    print(f"  Calendar: {len(calendar)} trading days")

    # ---- Step 1 ----
    print("\nStep 1: Building daily price/volume/market layer...")
    daily = _build_daily_layer(config)

    # Merge stock master
    merge_cols = [c for c in stock_master.columns
                  if c not in ("statco_current", "Markettype")
                  and c != config.stock_col
                  or c in ("list_date", "industry_csrc_2012", "is_financial", "stock_name_short", config.stock_col)]
    master_merge = stock_master[[c for c in merge_cols if c in stock_master.columns]]
    daily = daily.merge(
        master_merge, on=config.stock_col, how="left"
    )

    # Exclude financials
    if config.exclude_financials:
        n_before = daily[config.stock_col].nunique()
        daily = daily[daily["is_financial"] != True].copy()
        n_after = daily[config.stock_col].nunique()
        print(f"  Excluded {n_before - n_after} financial stocks")

    # Listing day filter: only drop stocks with KNOWN list_date that is too recent.
    # Stocks missing from TRD_Co (historical/delisted) are kept.
    if "list_date" in daily.columns:
        has_list_date = daily["list_date"].notna()
        daily["_days_listed"] = np.where(
            has_list_date,
            (pd.to_datetime(daily[config.date_col]) - daily["list_date"]).dt.days,
            np.nan,
        )
        n_before = daily[config.stock_col].nunique()
        too_new = has_list_date & (daily["_days_listed"] < config.min_listing_days)
        daily = daily[~too_new].copy()
        daily.drop(columns=["_days_listed"], inplace=True)
        n_after = daily[config.stock_col].nunique()
        print(f"  After min_listing_days filter: {n_before} → {n_after} stocks ({int(too_new.sum())} rows removed)")

    print(f"  Daily panel: {len(daily):,} rows, {daily[config.stock_col].nunique()} stocks")

    # ---- Step 1.5: total market cap from Capchg ----
    print("  Building total market cap from TRD_Capchg...")
    daily = _build_mktcap_total(daily, config)

    # ---- Step 2 ----
    print("\nStep 2: Loading quarterly financials...")
    financial = _load_financial(config)
    print(f"  Financial rows: {len(financial):,}, {financial[config.stock_col].nunique()} stocks")

    print("  Forward-filling to daily...")
    daily = _ffill_financial_to_daily(daily, financial, config)
    print(f"  After ffill: {len(daily):,} rows")

    # ---- Sanity checks ----
    _run_sanity_checks(daily, financial, calendar, config)

    # ---- Output ----
    print("\n--- Output ---")

    # Raw panel (2012+)
    raw_path = config.output_dir / "csmar_daily_raw_panel.parquet"
    daily.to_parquet(raw_path, index=False)
    print(f"Raw panel: {raw_path}")
    print(f"  rows={len(daily):,} stocks={daily[config.stock_col].nunique()} "
          f"dates={daily[config.date_col].min().date()}~{daily[config.date_col].max().date()}")

    # Factor-ready panel (2018+)
    factor_ready = daily[daily[config.date_col] >= config.model_start_date].copy()
    ready_path = config.output_dir / "csmar_factor_ready_panel.parquet"
    factor_ready.to_parquet(ready_path, index=False)
    print(f"Factor-ready panel: {ready_path}")
    print(f"  rows={len(factor_ready):,} stocks={factor_ready[config.stock_col].nunique()} "
          f"dates={factor_ready[config.date_col].min().date()}~{factor_ready[config.date_col].max().date()}")

    # Trajectory sample
    if daily[config.stock_col].nunique() > 0:
        sample_stock = daily[config.stock_col].value_counts().index[0]
        _sample_stock_trajectory(daily, config, sample_stock)

    print("\nDone.")


if __name__ == "__main__":
    main()
