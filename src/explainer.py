"""
Explainable AI (XAI) Module
NIFTY-50 Investment Intelligence Platform
Provides human-readable explanations for model predictions and portfolio recommendations.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
#  Manual SHAP-style permutation importance
# ─────────────────────────────────────────────────────────────────────────────

def permutation_importance(model, X: np.ndarray, y: np.ndarray,
                            feat_names: list, n_repeats: int = 5,
                            task: str = 'classification') -> pd.DataFrame:
    """
    Compute permutation-based feature importance.
    Measures how much model performance drops when a feature is shuffled.
    """
    from sklearn.metrics import accuracy_score, mean_squared_error

    if task == 'classification':
        base_score = accuracy_score(y, model.predict(X))
        score_fn   = lambda yp: accuracy_score(y, yp)
        better     = max
    else:
        base_score = -mean_squared_error(y, model.predict(X))
        score_fn   = lambda yp: -mean_squared_error(y, yp)
        better     = max

    results = []
    rng = np.random.default_rng(42)
    for i, feat in enumerate(feat_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, i])
            perm_score = score_fn(model.predict(X_perm))
            drops.append(base_score - perm_score)
        results.append({
            'Feature':    feat,
            'Importance': round(np.mean(drops), 6),
            'Std':        round(np.std(drops), 6),
        })

    df = pd.DataFrame(results).sort_values('Importance', ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Single-prediction explanation
# ─────────────────────────────────────────────────────────────────────────────

INDICATOR_DESCRIPTIONS = {
    'RSI_14':        ('RSI (momentum oscillator)', 'RSI > 70 = overbought, RSI < 30 = oversold'),
    'MACD':          ('MACD (trend momentum)',      'Positive MACD = bullish momentum, Negative = bearish'),
    'MACD_Hist':     ('MACD Histogram',             'Rising bars = strengthening trend'),
    'BB_Pct':        ('Bollinger Band %',           '> 0.8 = near upper band (overbought), < 0.2 = near lower (oversold)'),
    'BB_Width':      ('Bollinger Band Width',       'High width = high volatility regime'),
    'Volatility_20': ('20-day Volatility',          'Higher volatility = higher risk'),
    'MA_20':         ('20-day Moving Average',      'Price above MA20 = short-term uptrend'),
    'MA_50':         ('50-day Moving Average',      'Price above MA50 = medium-term uptrend'),
    'EMA_20':        ('20-day EMA',                 'Faster-reacting trend signal'),
    'Stoch_K':       ('Stochastic %K',              '> 80 = overbought zone, < 20 = oversold zone'),
    'Momentum_5':    ('5-day Momentum',             'Recent short-term price change'),
    'Momentum_20':   ('20-day Momentum',            'Medium-term price change direction'),
    'ATR_14':        ('ATR (volatility measure)',   'Higher ATR = larger daily price swings'),
    'Daily_Return':  ('Previous day return',        'Most recent price move'),
}


def explain_prediction(latest_row: pd.Series, prediction: dict,
                        importances: dict) -> dict:
    """
    Generate a human-readable explanation for a model prediction.
    """
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:8]

    reasons = []
    for feat, imp in top_feats:
        if feat not in latest_row.index or imp < 0.001:
            continue
        val = latest_row[feat]
        if pd.isna(val):
            continue
        desc, rule = INDICATOR_DESCRIPTIONS.get(feat, (feat, ''))

        # Interpret direction of signal
        bullish = False
        bearish = False

        if feat == 'RSI_14':
            if val < 35:
                bullish = True
                interpretation = f"RSI = {val:.1f} — oversold territory, potential reversal upward"
            elif val > 65:
                bearish = True
                interpretation = f"RSI = {val:.1f} — overbought territory, potential pullback"
            else:
                interpretation = f"RSI = {val:.1f} — neutral zone"
        elif feat == 'MACD':
            if val > 0:
                bullish = True
                interpretation = f"MACD = {val:.2f} — positive (bullish momentum)"
            else:
                bearish = True
                interpretation = f"MACD = {val:.2f} — negative (bearish momentum)"
        elif feat == 'MACD_Hist':
            if val > 0:
                bullish = True
                interpretation = f"MACD Histogram = {val:.2f} — rising trend strength"
            else:
                bearish = True
                interpretation = f"MACD Histogram = {val:.2f} — falling trend strength"
        elif feat == 'BB_Pct':
            if val < 0.25:
                bullish = True
                interpretation = f"BB% = {val:.2f} — price near lower band (oversold)"
            elif val > 0.75:
                bearish = True
                interpretation = f"BB% = {val:.2f} — price near upper band (overbought)"
            else:
                interpretation = f"BB% = {val:.2f} — price mid-range within bands"
        elif feat == 'Stoch_K':
            if val < 25:
                bullish = True
                interpretation = f"Stochastic %K = {val:.1f} — oversold"
            elif val > 75:
                bearish = True
                interpretation = f"Stochastic %K = {val:.1f} — overbought"
            else:
                interpretation = f"Stochastic %K = {val:.1f} — neutral"
        elif 'Momentum' in feat:
            if val > 0:
                bullish = True
                interpretation = f"{feat} = {val*100:.1f}% — positive momentum"
            else:
                bearish = True
                interpretation = f"{feat} = {val*100:.1f}% — negative momentum"
        elif feat == 'Volatility_20':
            interpretation = f"20d Volatility = {val*100:.1f}% — {'high' if val > 0.25 else 'moderate'} risk environment"
        else:
            interpretation = f"{desc} = {val:.3f} ({rule})"

        reasons.append({
            'Feature':        feat,
            'Description':    desc,
            'Value':          round(val, 4),
            'Interpretation': interpretation,
            'Importance_Pct': round(imp * 100, 1),
            'Signal':         'BULLISH' if bullish else ('BEARISH' if bearish else 'NEUTRAL'),
        })

    # Overall signal summary
    bullish_count = sum(1 for r in reasons if r['Signal'] == 'BULLISH')
    bearish_count = sum(1 for r in reasons if r['Signal'] == 'BEARISH')

    consensus = 'Mixed'
    if bullish_count >= bearish_count + 2:
        consensus = 'Mostly Bullish'
    elif bearish_count >= bullish_count + 2:
        consensus = 'Mostly Bearish'
    elif bullish_count > bearish_count:
        consensus = 'Slightly Bullish'
    elif bearish_count > bullish_count:
        consensus = 'Slightly Bearish'

    return {
        'Prediction':      prediction,
        'Reasons':         reasons,
        'Bullish_Signals': bullish_count,
        'Bearish_Signals': bearish_count,
        'Consensus':       consensus,
        'Summary':         _generate_narrative(prediction, reasons, consensus),
    }


def _generate_narrative(prediction: dict, reasons: list, consensus: str) -> str:
    """Generate a plain-English investment narrative."""
    signal    = prediction.get('Signal', 'NEUTRAL')
    direction = prediction.get('Direction', 'UP')
    prob      = prediction.get('Direction_Prob', 50)
    ret5d     = prediction.get('Pred_Return_5d', 0)
    conf      = prediction.get('Confidence', 'Low')

    lines = []
    lines.append(
        f"The model signals **{signal}** with {conf.lower()} confidence ({prob:.1f}% probability of upward move)."
    )
    lines.append(
        f"The predicted 5-day return is **{ret5d:+.2f}%**, suggesting a {'modest gain' if ret5d > 0 else 'potential decline'}."
    )

    top_reason = reasons[0] if reasons else None
    if top_reason:
        lines.append(
            f"The most influential factor is the **{top_reason['Description']}** "
            f"({top_reason['Importance_Pct']:.1f}% importance): {top_reason['Interpretation']}."
        )

    lines.append(
        f"Overall, {len(reasons)} key technical indicators show a **{consensus}** consensus."
    )

    return ' '.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolio explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_portfolio(portfolio_result: dict, risk_profiles: pd.DataFrame) -> dict:
    """Generate human-readable justification for each portfolio holding."""
    weights    = portfolio_result['Weights']
    profile    = portfolio_result['Profile']
    metrics    = portfolio_result['Metrics']
    risk_df    = risk_profiles.set_index('Symbol') if risk_profiles is not None else None

    explanations = []
    for sym, wt in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        entry = {'Symbol': sym, 'Weight_Pct': round(wt * 100, 1)}
        if risk_df is not None and sym in risk_df.index:
            row = risk_df.loc[sym]
            entry['Ann_Return']   = row.get('Ann_Return', None)
            entry['Ann_Vol']      = row.get('Ann_Volatility', None)
            entry['Sharpe']       = row.get('Sharpe_Ratio', None)
            entry['Risk_Tier']    = row.get('Risk_Tier', None)
            entry['Sector']       = row.get('Sector', None)

            # Justify inclusion
            reasons = []
            if profile == 'Conservative':
                if row.get('Ann_Volatility', 999) < 22:
                    reasons.append(f"low volatility ({row['Ann_Volatility']:.1f}% p.a.)")
                if row.get('Sharpe_Ratio', 0) > 0.5:
                    reasons.append(f"solid Sharpe ratio ({row['Sharpe_Ratio']:.2f})")
            elif profile == 'Balanced':
                if row.get('Sharpe_Ratio', 0) > 0.7:
                    reasons.append(f"strong risk-adjusted returns (Sharpe {row['Sharpe_Ratio']:.2f})")
            elif profile == 'Aggressive':
                if row.get('Ann_Return', 0) > 15:
                    reasons.append(f"high historical return ({row['Ann_Return']:.1f}% p.a.)")

            if row.get('Max_Drawdown', -100) > -25:
                reasons.append(f"controlled drawdown ({row['Max_Drawdown']:.1f}%)")

            entry['Justification'] = (', '.join(reasons) + '.').capitalize() if reasons else 'Contributes to portfolio diversification.'

        explanations.append(entry)

    return {
        'Profile':      profile,
        'Holdings':     explanations,
        'Summary_Metrics': metrics,
        'Portfolio_Narrative': _portfolio_narrative(profile, metrics, len(weights)),
    }


def _portfolio_narrative(profile: str, metrics: dict, n_holdings: int) -> str:
    ret  = metrics.get('Ann_Return', 0)
    vol  = metrics.get('Ann_Volatility', 0)
    sr   = metrics.get('Sharpe_Ratio', 0)
    mdd  = metrics.get('Max_Drawdown', 0)

    intro = {
        'Conservative': "This portfolio prioritises **capital preservation** by targeting low-volatility, stable stocks.",
        'Balanced':     "This portfolio targets **optimal risk-adjusted returns** using Sharpe ratio maximisation.",
        'Aggressive':   "This portfolio pursues **maximum growth** by concentrating in high-return stocks.",
    }.get(profile, "A diversified portfolio.")

    return (
        f"{intro} "
        f"Across {n_holdings} holdings, the strategy historically delivered an annualised return of "
        f"**{ret:.1f}%** with volatility of **{vol:.1f}%**, yielding a Sharpe ratio of **{sr:.2f}**. "
        f"The maximum drawdown experienced was **{mdd:.1f}%**, indicating the worst peak-to-trough "
        f"decline an investor would have faced."
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Technical signal summary
# ─────────────────────────────────────────────────────────────────────────────

def technical_signal_summary(df: pd.DataFrame) -> dict:
    """Return a dashboard-ready summary of current technical signals for a stock."""
    latest = df.sort_values('Date').iloc[-1]
    close  = latest['Close']

    signals = {}

    # Trend signals
    if 'MA_20' in df.columns and 'MA_50' in df.columns:
        signals['MA_Trend'] = {
            'value':  'Bullish' if close > latest.get('MA_50', close) else 'Bearish',
            'detail': f"Price ₹{close:.2f} vs MA50 ₹{latest.get('MA_50', 0):.2f}",
        }

    if 'MA_50' in df.columns and 'MA_200' in df.columns:
        ma50  = latest.get('MA_50', None)
        ma200 = latest.get('MA_200', None)
        if ma50 and ma200:
            signals['Golden_Death_Cross'] = {
                'value':  'Golden Cross' if ma50 > ma200 else 'Death Cross',
                'detail': f"MA50 ₹{ma50:.2f} vs MA200 ₹{ma200:.2f}",
            }

    # RSI
    if 'RSI_14' in df.columns:
        rsi = latest['RSI_14']
        signals['RSI'] = {
            'value':  'Oversold' if rsi < 30 else ('Overbought' if rsi > 70 else 'Neutral'),
            'detail': f"RSI = {rsi:.1f}",
        }

    # MACD
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        macd = latest['MACD']
        sig  = latest['MACD_Signal']
        signals['MACD'] = {
            'value':  'Bullish' if macd > sig else 'Bearish',
            'detail': f"MACD {macd:.2f} vs Signal {sig:.2f}",
        }

    # Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        bb_upper = latest['BB_Upper']
        bb_lower = latest['BB_Lower']
        bb_pct   = latest.get('BB_Pct', 0.5)
        signals['BB'] = {
            'value':  'Near Upper' if bb_pct > 0.8 else ('Near Lower' if bb_pct < 0.2 else 'Mid-Range'),
            'detail': f"BB [{bb_lower:.2f} – {bb_upper:.2f}], price at {bb_pct*100:.0f}%",
        }

    # Stochastic
    if 'Stoch_K' in df.columns:
        k = latest['Stoch_K']
        signals['Stochastic'] = {
            'value':  'Oversold' if k < 20 else ('Overbought' if k > 80 else 'Neutral'),
            'detail': f"Stoch %K = {k:.1f}",
        }

    # Volatility regime
    if 'Volatility_20' in df.columns:
        v = latest['Volatility_20']
        signals['Volatility_Regime'] = {
            'value':  'High' if v > 0.30 else ('Low' if v < 0.15 else 'Normal'),
            'detail': f"20d Ann. Vol = {v*100:.1f}%",
        }

    # Count bullish vs bearish
    bullish = sum(1 for s in signals.values() if s['value'] in ('Bullish', 'Golden Cross', 'Oversold'))
    bearish = sum(1 for s in signals.values() if s['value'] in ('Bearish', 'Death Cross', 'Overbought', 'Near Upper'))

    overall = 'BULLISH' if bullish > bearish else ('BEARISH' if bearish > bullish else 'NEUTRAL')

    return {
        'Signals':  signals,
        'Bullish':  bullish,
        'Bearish':  bearish,
        'Overall':  overall,
        'Latest_Close': close,
        'Date':         str(latest['Date'])[:10],
    }


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_stock, add_technical_indicators
    df = load_stock('RELIANCE')
    df = add_technical_indicators(df)
    summary = technical_signal_summary(df)
    print("=== Technical Signal Summary for RELIANCE ===")
    for k, v in summary['Signals'].items():
        print(f"  {k}: {v['value']} — {v['detail']}")
    print(f"  Overall: {summary['Overall']} ({summary['Bullish']} bullish / {summary['Bearish']} bearish)")
