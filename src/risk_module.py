"""
Risk Assessment Module
NIFTY-50 Investment Intelligence Platform
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

TRADING_DAYS = 252
RISK_FREE_RATE = 0.06   # 6% p.a. (approx Indian 10-yr Gsec yield)


# ─────────────────────────────────────────────────────────────────────────────
#  Single-stock risk metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stock_risk(df: pd.DataFrame, symbol: str) -> dict:
    """Compute a full risk profile for one stock."""
    df = df.copy().sort_values('Date')
    ret = df['Close'].pct_change().dropna()
    log_ret = np.log(df['Close'] / df['Close'].shift(1)).dropna()

    # Annualised returns & volatility
    ann_return = (1 + ret.mean()) ** TRADING_DAYS - 1
    ann_vol    = ret.std() * np.sqrt(TRADING_DAYS)
    daily_rf   = RISK_FREE_RATE / TRADING_DAYS

    # Sharpe ratio
    excess     = ret - daily_rf
    sharpe     = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS) if excess.std() > 0 else np.nan

    # Sortino ratio  (downside only)
    downside   = ret[ret < daily_rf] - daily_rf
    downside_std = downside.std() * np.sqrt(TRADING_DAYS)
    sortino    = (ann_return - RISK_FREE_RATE) / downside_std if downside_std > 0 else np.nan

    # Max Drawdown
    cum_ret    = (1 + ret).cumprod()
    roll_max   = cum_ret.cummax()
    drawdown   = (cum_ret - roll_max) / roll_max
    max_dd     = drawdown.min()

    # Calmar ratio
    calmar     = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    # VaR & CVaR (95%)
    var_95     = np.percentile(ret, 5)
    cvar_95    = ret[ret <= var_95].mean()

    # Beta vs NIFTY approximation  (using all NIFTY-50 avg return as proxy)
    market_proxy = ret   # placeholder; overridden in portfolio module

    # Skewness & Kurtosis
    skew  = float(stats.skew(ret))
    kurt  = float(stats.kurtosis(ret))

    # Rolling volatility (last 60 days)
    rolling_vol_60 = ret.rolling(60).std().iloc[-1] * np.sqrt(TRADING_DAYS)

    # Trend: % of days price above MA-200
    ma200 = df['Close'].rolling(200).mean()
    pct_above_ma200 = (df['Close'] > ma200).mean()

    return {
        'Symbol':           symbol,
        'Ann_Return':       round(ann_return * 100, 2),
        'Ann_Volatility':   round(ann_vol * 100, 2),
        'Sharpe_Ratio':     round(sharpe, 3),
        'Sortino_Ratio':    round(sortino, 3),
        'Max_Drawdown':     round(max_dd * 100, 2),
        'Calmar_Ratio':     round(calmar, 3),
        'VaR_95':           round(var_95 * 100, 2),
        'CVaR_95':          round(cvar_95 * 100, 2),
        'Skewness':         round(skew, 3),
        'Kurtosis':         round(kurt, 3),
        'Rolling_Vol_60d':  round(rolling_vol_60 * 100, 2),
        'Pct_Above_MA200':  round(pct_above_ma200 * 100, 2),
    }


def compute_all_risk_profiles(all_df: pd.DataFrame) -> pd.DataFrame:
    """Compute risk metrics for all stocks in the combined dataframe."""
    results = []
    for symbol, gdf in all_df.groupby('Symbol'):
        try:
            r = compute_stock_risk(gdf, symbol)
            r['Sector'] = gdf['Sector'].iloc[0]
            r['Name']   = gdf['Name'].iloc[0]
            results.append(r)
        except Exception as e:
            print(f"  Warning: risk calc failed for {symbol}: {e}")
    df_risk = pd.DataFrame(results)
    # Risk tier classification
    df_risk['Risk_Tier'] = df_risk['Ann_Volatility'].apply(_classify_risk)
    return df_risk.sort_values('Sharpe_Ratio', ascending=False).reset_index(drop=True)


def _classify_risk(ann_vol: float) -> str:
    if ann_vol < 20:
        return 'Low'
    elif ann_vol < 35:
        return 'Medium'
    else:
        return 'High'


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolio risk
# ─────────────────────────────────────────────────────────────────────────────

def portfolio_risk(weights: np.ndarray, returns: pd.DataFrame) -> dict:
    """
    Compute risk metrics for a weighted portfolio.
    weights : array of floats summing to 1, aligned with returns.columns
    returns : daily returns matrix (dates × symbols)
    """
    w       = np.array(weights)
    port_ret = returns @ w
    port_ret = port_ret.dropna()

    ann_return = (1 + port_ret.mean()) ** TRADING_DAYS - 1
    ann_vol    = port_ret.std() * np.sqrt(TRADING_DAYS)
    daily_rf   = RISK_FREE_RATE / TRADING_DAYS
    excess     = port_ret - daily_rf
    sharpe     = excess.mean() / excess.std() * np.sqrt(TRADING_DAYS)

    downside   = port_ret[port_ret < daily_rf] - daily_rf
    downside_std = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 1e-9
    sortino    = (ann_return - RISK_FREE_RATE) / downside_std

    cum        = (1 + port_ret).cumprod()
    roll_max   = cum.cummax()
    max_dd     = ((cum - roll_max) / roll_max).min()
    calmar     = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    var_95     = np.percentile(port_ret, 5)
    cvar_95    = port_ret[port_ret <= var_95].mean()

    cov_matrix = returns.cov() * TRADING_DAYS
    port_variance = float(w @ cov_matrix.values @ w)

    return {
        'Ann_Return':     round(ann_return * 100, 2),
        'Ann_Volatility': round(ann_vol * 100, 2),
        'Sharpe_Ratio':   round(sharpe, 3),
        'Sortino_Ratio':  round(sortino, 3),
        'Max_Drawdown':   round(max_dd * 100, 2),
        'Calmar_Ratio':   round(calmar, 3),
        'VaR_95':         round(var_95 * 100, 2),
        'CVaR_95':        round(cvar_95 * 100, 2),
        'Portfolio_Variance': round(port_variance * 100, 4),
    }


def rolling_portfolio_metrics(weights: np.ndarray, returns: pd.DataFrame,
                               window: int = 252) -> pd.DataFrame:
    """Rolling Sharpe and volatility over a sliding window."""
    w       = np.array(weights)
    port_ret = (returns @ w).dropna()
    daily_rf = RISK_FREE_RATE / TRADING_DAYS

    roll_vol   = port_ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
    roll_excess = port_ret - daily_rf
    roll_sharpe = (roll_excess.rolling(window).mean() /
                   roll_excess.rolling(window).std() * np.sqrt(TRADING_DAYS))

    cum = (1 + port_ret).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max) / roll_max

    return pd.DataFrame({
        'Rolling_Vol':    roll_vol,
        'Rolling_Sharpe': roll_sharpe,
        'Drawdown':       drawdown,
    })


# ─────────────────────────────────────────────────────────────────────────────
#  Anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag days with abnormal returns, volume spikes, or volatility jumps.
    Returns a dataframe of anomalous events.
    """
    df = df.copy().sort_values('Date')
    ret = df['Close'].pct_change()

    ret_mean = ret.rolling(60, min_periods=20).mean()
    ret_std  = ret.rolling(60, min_periods=20).std()
    z_ret    = (ret - ret_mean) / (ret_std + 1e-9)

    anomalies = []

    # Return anomalies
    mask = z_ret.abs() > z_threshold
    for idx in df[mask].index:
        row = df.loc[idx]
        z   = z_ret.loc[idx]
        anomalies.append({
            'Date':   row['Date'],
            'Symbol': row.get('Symbol', ''),
            'Type':   'Return Spike' if z > 0 else 'Return Crash',
            'Close':  row['Close'],
            'Return_Pct': round(ret.loc[idx] * 100, 2),
            'Z_Score': round(z, 2),
            'Severity': 'High' if abs(z) > 4.5 else 'Medium',
        })

    # Volume spikes (if available)
    if 'Volume' in df.columns and df['Volume'].notna().sum() > 30:
        vol_mean = df['Volume'].rolling(20, min_periods=10).mean()
        vol_ratio = df['Volume'] / (vol_mean + 1)
        for idx in df[vol_ratio > 3].index:
            row = df.loc[idx]
            anomalies.append({
                'Date':    row['Date'],
                'Symbol':  row.get('Symbol', ''),
                'Type':    'Volume Spike',
                'Close':   row['Close'],
                'Return_Pct': round(ret.loc[idx] * 100 if idx in ret.index else 0, 2),
                'Z_Score': round(vol_ratio.loc[idx], 2),
                'Severity': 'High' if vol_ratio.loc[idx] > 5 else 'Medium',
            })

    if not anomalies:
        return pd.DataFrame()
    adf = pd.DataFrame(anomalies).drop_duplicates(subset=['Date', 'Type'])
    return adf.sort_values('Date').reset_index(drop=True)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_stock, add_technical_indicators
    df = load_stock('TCS')
    df = add_technical_indicators(df)
    risk = compute_stock_risk(df, 'TCS')
    for k, v in risk.items():
        print(f"  {k}: {v}")
    anomalies = detect_anomalies(df)
    print(f"\nAnomalies detected: {len(anomalies)}")
    print(anomalies.head())
