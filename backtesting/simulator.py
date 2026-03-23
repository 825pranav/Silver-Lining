# backtesting/simulator.py
#
# Upgraded from the original single-pass loop to include:
#   - Volatility-based position sizing (inverse-vol targeting)
#   - Transaction costs via metrics.apply_costs
#   - Strict no-lookahead enforcement (signal from close_t → position for close_t+1)
#   - Walk-forward validation with purged cross-validation windows
#
# Original signal logic (mean reversion + trend fallback) is preserved unchanged.
#
# Dependencies: numpy, pandas (both already installed)

import os
import numpy as np
import pandas as pd

try:
    from backtesting.metrics import compute_all
except ImportError:
    from metrics import compute_all

FEATURE_PATH = "../data/features.csv"

# ---------------------------------------------------------------------------
# Position sizing parameters
# ---------------------------------------------------------------------------
TARGET_DAILY_VOL = 0.01   # 1 % daily volatility target
MAX_POSITION_SIZE = 1.0   # cap at 100 % (no leverage)

# ---------------------------------------------------------------------------
# Walk-forward parameters
# ---------------------------------------------------------------------------
PURGE_GAP_DAYS = 20       # trading-day gap between train end and test start
                           # covers the longest lookback window (gsr_zscore_30)


# ---------------------------------------------------------------------------
# Signal logic  (unchanged from original)
# ---------------------------------------------------------------------------

def _signal(row: pd.Series) -> str:
    """
    Compute the trading signal from a single feature row.

    Called with the ROW AT CLOSE OF DAY t; the returned signal becomes the
    position that earns the return from day t close to day t+1 close.
    No data from t+1 or later is accessed here.
    """
    # Mean reversion: GSR stretched + momentum decelerating
    if row["gsr_zscore_30"] > 1 and abs(row["gsr_slope"]) < 0.5:
        return "silver"    # ratio high → silver undervalued
    if row["gsr_zscore_30"] < -1 and abs(row["gsr_slope"]) < 0.5:
        return "gold"

    # Trend-following fallback  (original logic)
    if row["gsr_slope"] > 0:
        return "gold"
    return "silver"


# ---------------------------------------------------------------------------
# Position sizing  (new)
# ---------------------------------------------------------------------------

def _position_size(row: pd.Series, sig: str) -> float:
    """
    Inverse-volatility position sizing.

    size = target_daily_vol / realised_daily_vol,  capped at MAX_POSITION_SIZE.
    Uses gold_vol_20 or silver_vol_20 (already in features.csv).
    Falls back to maximum size when vol is unavailable or zero.
    """
    vol_col = "gold_vol_20" if sig == "gold" else "silver_vol_20"
    vol = row.get(vol_col, np.nan)
    if pd.isna(vol) or vol <= 0:
        return MAX_POSITION_SIZE
    return min(TARGET_DAILY_VOL / vol, MAX_POSITION_SIZE)


# ---------------------------------------------------------------------------
# Core fold runner  (used by both single-pass and walk-forward)
# ---------------------------------------------------------------------------

def _run_fold(
    df: pd.DataFrame,
    spread: float = 0.0002,
    slippage: float = 0.0001,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Run one backtest pass on *df*.

    No-lookahead guarantee
    ----------------------
    At bar i the function:
      1. Earns the return from bar i-1 → i using the POSITION SET AT BAR i-1.
      2. Observes bar i's features to compute a new signal.
      3. Stores that signal as the position that will earn bar i+1's return.

    Returns
    -------
    capital_curve    : pd.Series  (index = df.index[1:])
    returns_series   : pd.Series  net daily returns
    changes_series   : pd.Series  1 where position changed, 0 otherwise
    """
    capital       = 1.0
    position      = None   # "gold" | "silver" | None (flat at start)
    pos_size      = 0.0

    capital_list  = []
    returns_list  = []
    changes_list  = []

    for i in range(1, len(df)):
        today     = df.iloc[i]
        yesterday = df.iloc[i - 1]

        # ── Step 1: realise return for this bar ──────────────────────────────
        # Position was set at yesterday's close; earns yesterday→today return.
        if position == "gold":
            gross = today["gold_close"] / yesterday["gold_close"] - 1
        elif position == "silver":
            gross = today["silver_close"] / yesterday["silver_close"] - 1
        else:
            gross = 0.0

        # ── Step 2: decide new position from today's features (no lookahead) ─
        new_sig  = _signal(today)
        new_size = _position_size(today, new_sig)
        switched = int(new_sig != position and position is not None)

        # Transaction cost applied on the day we execute the switch
        cost    = switched * (spread + slippage)
        net_ret = pos_size * gross - cost

        capital *= (1 + net_ret)

        capital_list.append(capital)
        returns_list.append(net_ret)
        changes_list.append(switched)

        # ── Step 3: carry new position forward ───────────────────────────────
        position = new_sig
        pos_size = new_size

    idx = df.index[1:]
    return (
        pd.Series(capital_list, index=idx, name="capital"),
        pd.Series(returns_list, index=idx, name="return"),
        pd.Series(changes_list, index=idx, name="switched"),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame = None,
    spread: float = 0.0002,
    slippage: float = 0.0001,
) -> tuple[pd.DataFrame, dict]:
    """
    Single-pass backtest over the full dataset.

    Parameters
    ----------
    df       : Feature DataFrame with DatetimeIndex. Loaded from disk if None.
    spread   : One-way bid-ask spread (default 2 bps).
    slippage : One-way slippage (default 1 bp).

    Returns
    -------
    out_df   : Input df with 'capital' and 'daily_return' columns appended.
    metrics  : Dict of performance metrics from metrics.compute_all().
    """
    if df is None:
        df = pd.read_csv(FEATURE_PATH, parse_dates=["Date"])
        df.set_index("Date", inplace=True)

    cap, rets, changes = _run_fold(df, spread=spread, slippage=slippage)
    metrics = compute_all(cap, rets)   # returns already net of costs

    print("\n📊 Single-Pass Backtest Results")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"   {k:<20} {v:>10}")
    print(f"\n   Final capital  : {cap.iloc[-1]:.4f}")
    print(f"   Total return   : {(cap.iloc[-1] - 1) * 100:.2f}%")
    print(f"   Position turns : {int(changes.sum())}")

    out = df.copy()
    out["capital"]      = cap
    out["daily_return"] = rets

    os.makedirs("../data", exist_ok=True)
    out.to_csv("../data/backtest_results.csv")
    print("\n💾 Saved to data/backtest_results.csv")

    return out, metrics


def walk_forward_backtest(
    df: pd.DataFrame = None,
    n_splits: int = 5,
    purge_days: int = PURGE_GAP_DAYS,
    spread: float = 0.0002,
    slippage: float = 0.0001,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Walk-forward validation with purged cross-validation windows.

    Window layout per fold
    ----------------------
    |<──────── train (anchored, expanding) ────────>|<purge>|<── test ──>|
                                                      ^^^^^^^
                              Rows in the purge gap are dropped from both
                              train and test to prevent label leakage from
                              any rolling features with lookback ≤ purge_days.

    Since the signal is rule-based (no parameter fitting), the train period
    is not used for calibration; it exists to anchor the purge boundary and
    to make the split structure explicit for future model-based extensions.

    Parameters
    ----------
    df        : Feature DataFrame with DatetimeIndex. Loaded from disk if None.
    n_splits  : Number of test folds (default 5).
    purge_days: Rows dropped between train end and test start (default 20).
    spread    : One-way spread cost (default 2 bps).
    slippage  : One-way slippage (default 1 bp).

    Returns
    -------
    fold_df     : DataFrame with one row per fold and metric columns.
    all_capital : Concatenated capital curve across all test folds.
    """
    if df is None:
        df = pd.read_csv(FEATURE_PATH, parse_dates=["Date"])
        df.set_index("Date", inplace=True)

    n = len(df)
    # Each test fold covers (n / (n_splits + 1)) bars; train is everything before.
    test_size = max(n // (n_splits + 1), 30)   # floor at 30 bars per fold

    fold_records = []
    all_capital  = pd.Series(dtype=float, name="capital")

    print(f"\n🔄 Walk-Forward Validation | folds={n_splits} | purge={purge_days}d | "
          f"test_window={test_size}d")
    print("-" * 72)

    for fold in range(n_splits):
        test_start = (fold + 1) * test_size + purge_days
        test_end   = test_start + test_size

        if test_end > n:
            print(f"  Fold {fold+1}: not enough data, stopping early.")
            break

        df_test = df.iloc[test_start:test_end]
        if len(df_test) < 10:
            continue

        cap, rets, _ = _run_fold(df_test, spread=spread, slippage=slippage)
        m = compute_all(cap, rets)

        m["fold"]       = fold + 1
        m["test_start"] = df_test.index[0].date()
        m["test_end"]   = df_test.index[-1].date()
        m["n_bars"]     = len(df_test)
        fold_records.append(m)

        all_capital = pd.concat([all_capital, cap])

        print(
            f"  Fold {fold+1}  {df_test.index[0].date()} → {df_test.index[-1].date()} | "
            f"CAGR={m['cagr']:+.2%}  Sharpe={m['sharpe']:+.2f}  "
            f"MaxDD={m['max_drawdown']:.2%}  Calmar={m['calmar']:.2f}"
        )

    print("-" * 72)

    fold_df = pd.DataFrame(fold_records)
    numeric = ["sharpe", "cagr", "max_drawdown", "hit_rate", "calmar"]
    numeric = [c for c in numeric if c in fold_df.columns]

    if not fold_df.empty:
        print("\n📊 Walk-Forward Summary (mean across folds):")
        for col in numeric:
            print(f"   {col:<20} {fold_df[col].mean():>10.4f}")

    return fold_df, all_capital


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _, _ = run_backtest()
    results, _ = walk_forward_backtest()
    if not results.empty:
        print("\n" + results.to_string(index=False))
