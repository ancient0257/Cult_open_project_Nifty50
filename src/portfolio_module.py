"""
Portfolio Construction Module
NIFTY-50 Investment Intelligence Platform
Uses mean-variance optimisation (Markowitz) via scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

TRADING_DAYS = 252
RISK_FREE_RATE = 0.06


# ─────────────────────────────────────────────────────────────────────────────
#  Core optimisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _port_return(w, mean_returns):
    return np.dot(w, mean_returns) * TRADING_DAYS

def _port_volatility(w, cov_matrix):
    return np.sqrt(w @ cov_matrix @ w * TRADING_DAYS)

def _neg_sharpe(w, mean_returns, cov_matrix, rf=RISK_FREE_RATE):
    ret = _port_return(w, mean_returns)
    vol = _port_volatility(w, cov_matrix)
    return -(ret - rf) / vol

def _neg_sortino(w, returns_df, rf=RISK_FREE_RATE):
    port_ret = returns_df @ w
    ann_ret  = (1 + port_ret.mean()) ** TRADING_DAYS - 1
    daily_rf = rf / TRADING_DAYS
    downside = port_ret[port_ret < daily_rf] - daily_rf
    ds_std   = downside.std() * np.sqrt(TRADING_DAYS) + 1e-9
    return -(ann_ret - rf) / ds_std

def _portfolio_volatility_only(w, cov_matrix):
    return _port_volatility(w, cov_matrix)

def _neg_return(w, mean_returns):
    return -_port_return(w, mean_returns)


def _optimise(objective, returns_df: pd.DataFrame,
              bounds_per_asset=(0.0, 0.40),
              extra_constraints=None,
              objective_kwargs=None) -> np.ndarray:
    """Generic optimiser wrapper."""
    n = returns_df.shape[1]
    mean_ret = returns_df.mean().values
    cov      = returns_df.cov().values

    objective_kwargs = objective_kwargs or {}
    constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
    if extra_constraints:
        constraints.extend(extra_constraints)

    bounds = [bounds_per_asset] * n
    w0     = np.ones(n) / n

    result = minimize(
        objective, w0,
        args=tuple(objective_kwargs.values()),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    if not result.success:
        return w0  # fallback to equal weight
    return result.x


# ─────────────────────────────────────────────────────────────────────────────
#  Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

def efficient_frontier(returns_df: pd.DataFrame, n_points: int = 50):
    """Compute the efficient frontier by sweeping target return levels."""
    n = returns_df.shape[1]
    mean_ret = returns_df.mean().values * TRADING_DAYS
    cov      = returns_df.cov().values * TRADING_DAYS
    w0       = np.ones(n) / n

    target_returns = np.linspace(mean_ret.min(), mean_ret.max(), n_points)
    frontier = []
    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1},
            {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, mean_ret) - t},
        ]
        res = minimize(
            lambda w: w @ cov @ w,
            w0, method='SLSQP',
            bounds=[(0, 0.40)] * n,
            constraints=constraints,
            options={'ftol': 1e-10, 'maxiter': 500}
        )
        if res.success:
            vol = np.sqrt(res.fun)
            sr  = (target - RISK_FREE_RATE) / (vol + 1e-9)
            frontier.append({'Return': target * 100, 'Volatility': vol * 100, 'Sharpe': sr})

    return pd.DataFrame(frontier)


# ─────────────────────────────────────────────────────────────────────────────
#  Three investor profiles
# ─────────────────────────────────────────────────────────────────────────────

def build_conservative_portfolio(returns_df: pd.DataFrame,
                                  risk_profiles: pd.DataFrame = None) -> dict:
    """
    Conservative: minimise volatility, prefer low-volatility / high-Sharpe stocks.
    Max 15% per stock. Sector diversification constraint.
    """
    # Pre-filter: prefer low-volatility stocks if risk_profiles provided
    if risk_profiles is not None:
        low_vol = risk_profiles.nsmallest(30, 'Ann_Volatility')['Symbol'].tolist()
        cols = [c for c in returns_df.columns if c in low_vol]
        if len(cols) >= 10:
            returns_df = returns_df[cols]

    n = returns_df.shape[1]
    cov = returns_df.cov().values
    w = _optimise(
        lambda w, cov=cov: _portfolio_volatility_only(w, cov),
        returns_df,
        bounds_per_asset=(0.0, 0.15),
        objective_kwargs={'cov_matrix': cov}
    )
    return _build_result(w, returns_df, 'Conservative',
                         'Minimum Volatility — capital preservation, stable returns', risk_profiles)


def build_balanced_portfolio(returns_df: pd.DataFrame,
                              risk_profiles: pd.DataFrame = None) -> dict:
    """
    Balanced: maximise Sharpe ratio (best risk-adjusted returns).
    Max 20% per stock.
    """
    mean_ret = returns_df.mean().values
    cov      = returns_df.cov().values
    w = _optimise(
        _neg_sharpe,
        returns_df,
        bounds_per_asset=(0.0, 0.20),
        objective_kwargs={'mean_returns': mean_ret, 'cov_matrix': cov}
    )
    return _build_result(w, returns_df, 'Balanced',
                         'Maximum Sharpe Ratio — best risk-adjusted return', risk_profiles)


def build_aggressive_portfolio(returns_df: pd.DataFrame,
                                risk_profiles: pd.DataFrame = None) -> dict:
    """
    Aggressive: maximise return (momentum-biased), higher concentration allowed.
    Max 25% per stock. Uses top-momentum stocks.
    """
    # Focus on high-momentum stocks
    if risk_profiles is not None:
        high_ret = risk_profiles.nlargest(25, 'Ann_Return')['Symbol'].tolist()
        cols = [c for c in returns_df.columns if c in high_ret]
        if len(cols) >= 8:
            returns_df = returns_df[cols]

    mean_ret = returns_df.mean().values
    w = _optimise(
        _neg_return,
        returns_df,
        bounds_per_asset=(0.02, 0.25),
        objective_kwargs={'mean_returns': mean_ret}
    )
    return _build_result(w, returns_df, 'Aggressive',
                         'Maximum Return — growth-oriented, higher risk tolerance', risk_profiles)


def _build_result(weights: np.ndarray, returns_df: pd.DataFrame,
                   profile: str, description: str,
                   risk_profiles: pd.DataFrame = None) -> dict:
    """Package weights + metrics into a clean result dict."""
    symbols = returns_df.columns.tolist()
    w_series = pd.Series(weights, index=symbols)
    # Keep only meaningful allocations
    w_series = w_series[w_series > 0.005].sort_values(ascending=False)
    w_series = w_series / w_series.sum()

    # Portfolio metrics
    sub_ret = returns_df[w_series.index]
    from risk_module import portfolio_risk
    metrics = portfolio_risk(w_series.values, sub_ret)

    # Add sector breakdown
    sector_alloc = {}
    if risk_profiles is not None:
        sym_sec = risk_profiles.set_index('Symbol')['Sector'].to_dict()
        for sym, wt in w_series.items():
            sec = sym_sec.get(sym, 'Unknown')
            sector_alloc[sec] = sector_alloc.get(sec, 0) + wt

    return {
        'Profile':      profile,
        'Description':  description,
        'Weights':      w_series.round(4).to_dict(),
        'Metrics':      metrics,
        'Sector_Alloc': {k: round(v * 100, 1) for k, v in sector_alloc.items()},
        'N_Holdings':   len(w_series),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Equal-weight benchmark
# ─────────────────────────────────────────────────────────────────────────────

def build_equal_weight_portfolio(returns_df: pd.DataFrame,
                                  risk_profiles: pd.DataFrame = None) -> dict:
    n = returns_df.shape[1]
    w = np.ones(n) / n
    return _build_result(w, returns_df, 'Equal Weight',
                         '1/N equal weight — naive diversification benchmark', risk_profiles)


# ─────────────────────────────────────────────────────────────────────────────
#  Backtesting
# ─────────────────────────────────────────────────────────────────────────────

def backtest_portfolio(weights_dict: dict, prices_df: pd.DataFrame,
                        rebalance_freq: str = 'Q') -> pd.DataFrame:
    """
    Simple backtest with optional periodic rebalancing.
    weights_dict : {symbol: weight}
    prices_df    : wide price dataframe (dates × symbols)
    rebalance_freq: 'D' (daily), 'M' (monthly), 'Q' (quarterly)
    """
    syms   = [s for s in weights_dict if s in prices_df.columns]
    w      = np.array([weights_dict[s] for s in syms])
    w      = w / w.sum()
    prices = prices_df[syms].dropna()

    portfolio_values = []
    dates = prices.index.tolist()
    port_val = 100.0  # start at 100

    # Build rebalance dates
    freq_map = {'Q': 'QE', 'Y': 'YE', 'A': 'YE', 'M': 'ME', 'BM': 'BME'}
    rebalance_freq = freq_map.get(rebalance_freq, rebalance_freq)
    if rebalance_freq == 'D':
        rebalance_dates = set(dates)
    else:
        tmp = pd.Series(dates, index=dates)
        rebalance_dates = set(tmp.resample(rebalance_freq).first().values)

    holdings = w * port_val / prices.iloc[0].values  # shares

    for i, date in enumerate(dates):
        row = prices.iloc[i].values
        port_val = (holdings * row).sum()
        portfolio_values.append({'Date': date, 'Portfolio_Value': port_val})

        if date in rebalance_dates and i < len(dates) - 1:
            holdings = w * port_val / prices.iloc[i].values  # rebalance

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result['Return'] = result['Portfolio_Value'].pct_change()
    result['Cumulative_Return'] = result['Portfolio_Value'] / 100 - 1
    roll_max = result['Portfolio_Value'].cummax()
    result['Drawdown'] = (result['Portfolio_Value'] - roll_max) / roll_max
    return result


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import build_returns_matrix
    prices, returns = build_returns_matrix()
    returns_clean = returns.dropna(axis=1, thresh=int(len(returns)*0.7))
    returns_clean = returns_clean.fillna(0)

    balanced = build_balanced_portfolio(returns_clean)
    print("=== Balanced Portfolio ===")
    print("Holdings:", list(balanced['Weights'].keys())[:8])
    print("Metrics:", balanced['Metrics'])
    print("Sectors:", balanced['Sector_Alloc'])
