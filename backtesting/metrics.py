# backtesting/metrics.py
#
# Portfolio performance metrics.
# All functions accept pandas Series; compute_all() is the main entry point.
#
# Dependencies: numpy, pandas (both already installed)

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def apply_costs(
    returns: pd.Series,
    position_changes: pd.Series,
    spread: float = 0.0002,
    slippage: float = 0.0001,
) -> pd.Series:
    """
    Deduct one-way spread + slippage on every bar where the position changed.

    Parameters
    ----------
    returns          : Gross daily return series.
    position_changes : Integer series (1 = position switched, 0 = held).
    spread           : One-way bid-ask spread (default 2 bps).
    slippage         : One-way market-impact slippage (default 1 bp).

    Returns
    -------
    Net return series with transaction costs removed.
    """
    cost_per_turn = spread + slippage
    return returns - position_changes * cost_per_turn


def sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0)."""
    std = returns.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((returns.mean() / std) * np.sqrt(periods_per_year))


def cagr(capital_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Compound Annual Growth Rate."""
    n = len(capital_curve)
    if n < 2:
        return 0.0
    total = capital_curve.iloc[-1] / capital_curve.iloc[0]
    if total <= 0:
        return -1.0
    years = n / periods_per_year
    return float(total ** (1.0 / years) - 1.0)


def max_drawdown(capital_curve: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative value, e.g. -0.25 = -25%)."""
    rolling_peak = capital_curve.cummax()
    dd = (capital_curve - rolling_peak) / rolling_peak
    return float(dd.min())


def hit_rate(returns: pd.Series) -> float:
    """Fraction of non-zero periods with a positive return."""
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def calmar(cagr_val: float, max_dd_val: float) -> float:
    """Calmar ratio = CAGR / |max drawdown|. Returns 0 if drawdown is zero."""
    if max_dd_val == 0 or np.isnan(max_dd_val):
        return 0.0
    return float(cagr_val / abs(max_dd_val))


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_all(
    capital_curve: pd.Series,
    returns: pd.Series,
    position_changes: pd.Series = None,
    spread: float = 0.0002,
    slippage: float = 0.0001,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute all performance metrics in one call.

    If *position_changes* is supplied the function treats *returns* as gross
    returns and applies transaction costs before computing metrics.  Pass
    already-net returns and omit *position_changes* to skip cost adjustment.

    Parameters
    ----------
    capital_curve    : Portfolio value series starting at 1.0.
    returns          : Daily return series (gross or net).
    position_changes : Optional int series (1 = switched, 0 = held).
    spread           : One-way spread cost (default 2 bps).
    slippage         : One-way slippage cost (default 1 bp).
    periods_per_year : Trading days per year (default 252).

    Returns
    -------
    dict with keys: sharpe, cagr, max_drawdown, hit_rate, calmar.
    """
    if position_changes is not None:
        returns = apply_costs(returns, position_changes, spread=spread, slippage=slippage)
        capital_curve = (1 + returns).cumprod()

    cagr_val = cagr(capital_curve, periods_per_year)
    max_dd   = max_drawdown(capital_curve)

    return {
        "sharpe":       round(sharpe(returns, periods_per_year), 4),
        "cagr":         round(cagr_val, 4),
        "max_drawdown": round(max_dd, 4),
        "hit_rate":     round(hit_rate(returns), 4),
        "calmar":       round(calmar(cagr_val, max_dd), 4),
    }
