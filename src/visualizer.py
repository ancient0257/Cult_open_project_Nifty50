"""
Visualisation Module
NIFTY-50 Investment Intelligence Platform
Generates all charts using matplotlib/seaborn (no plotly dependency).
Saves PNGs to outputs/ directory for embedding in the Streamlit app.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
DARK_BG   = '#0d1117'
CARD_BG   = '#161b22'
ACCENT1   = '#58a6ff'   # blue
ACCENT2   = '#3fb950'   # green
ACCENT3   = '#f85149'   # red
ACCENT4   = '#d2a8ff'   # purple
GOLD      = '#e3b341'
GRID_COL  = '#30363d'
TEXT_COL  = '#e6edf3'
MUTED     = '#8b949e'

def _apply_dark_style(fig, axes=None):
    fig.patch.set_facecolor(DARK_BG)
    if axes is None:
        return
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)
        ax.grid(True, color=GRID_COL, linewidth=0.5, alpha=0.6)


# ─────────────────────────────────────────────────────────────────────────────
#  1. Candlestick + indicators dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_stock_dashboard(df: pd.DataFrame, symbol: str, name: str = '',
                          last_n: int = 252, save: bool = True) -> str:
    df = df.sort_values('Date').tail(last_n).reset_index(drop=True)
    dates = pd.to_datetime(df['Date'])

    fig = plt.figure(figsize=(16, 12), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(4, 1, hspace=0.05,
                             height_ratios=[4, 1.2, 1.2, 1.2])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    axes = [ax1, ax2, ax3, ax4]
    _apply_dark_style(fig, axes)

    # ── Price + MAs + BBands ─────────────────────────────────────────
    ax1.fill_between(dates, df['BB_Upper'], df['BB_Lower'],
                     alpha=0.08, color=ACCENT1, label='BB Band')
    ax1.plot(dates, df['BB_Upper'], color=ACCENT1, lw=0.6, alpha=0.5)
    ax1.plot(dates, df['BB_Lower'], color=ACCENT1, lw=0.6, alpha=0.5)
    ax1.plot(dates, df['MA_20'],  color=GOLD,    lw=1.2, label='MA20', alpha=0.9)
    ax1.plot(dates, df['MA_50'],  color=ACCENT4, lw=1.2, label='MA50', alpha=0.9)
    ax1.plot(dates, df['MA_200'] if 'MA_200' in df.columns else df['MA_50'],
             color=ACCENT3, lw=1.2, label='MA200', alpha=0.9, linestyle='--')

    # Price bars coloured by day return
    up   = df['Close'] >= df['Open']
    for i, (_, row) in enumerate(df.iterrows()):
        col = ACCENT2 if up.iloc[i] else ACCENT3
        ax1.plot([dates.iloc[i], dates.iloc[i]],
                 [row['Low'], row['High']], color=col, lw=0.6, alpha=0.8)
        ax1.bar(dates.iloc[i], row['Close'] - row['Open'],
                bottom=row['Open'], color=col, width=0.6, alpha=0.7)

    ax1.set_title(f"{name or symbol}  ({symbol})  —  Price Chart & Indicators",
                  color=TEXT_COL, fontsize=14, fontweight='bold', pad=10)
    ax1.set_ylabel('Price (₹)', color=MUTED)
    leg = ax1.legend(loc='upper left', fontsize=8, fancybox=True,
                     framealpha=0.3, labelcolor=TEXT_COL)
    leg.get_frame().set_facecolor(CARD_BG)

    # ── Volume ────────────────────────────────────────────────────────
    if 'Volume' in df.columns and df['Volume'].notna().sum() > 10:
        colors = [ACCENT2 if u else ACCENT3 for u in up]
        ax2.bar(dates, df['Volume'] / 1e6, color=colors, alpha=0.7, width=0.6)
        ax2.set_ylabel('Vol (M)', color=MUTED, fontsize=8)
    else:
        ax2.set_visible(False)

    # ── RSI ────────────────────────────────────────────────────────────
    ax3.plot(dates, df['RSI_14'], color=ACCENT4, lw=1.2)
    ax3.axhline(70, color=ACCENT3, lw=0.8, linestyle='--', alpha=0.7)
    ax3.axhline(30, color=ACCENT2, lw=0.8, linestyle='--', alpha=0.7)
    ax3.fill_between(dates, df['RSI_14'], 70,
                     where=df['RSI_14'] >= 70, alpha=0.2, color=ACCENT3)
    ax3.fill_between(dates, df['RSI_14'], 30,
                     where=df['RSI_14'] <= 30, alpha=0.2, color=ACCENT2)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel('RSI', color=MUTED, fontsize=8)

    # ── MACD ────────────────────────────────────────────────────────────
    ax4.plot(dates, df['MACD'],        color=ACCENT1,  lw=1.2, label='MACD')
    ax4.plot(dates, df['MACD_Signal'], color=ACCENT3,  lw=1.2, label='Signal')
    hist_pos = df['MACD_Hist'].clip(lower=0)
    hist_neg = df['MACD_Hist'].clip(upper=0)
    ax4.bar(dates, hist_pos, color=ACCENT2, alpha=0.6, width=0.6)
    ax4.bar(dates, hist_neg, color=ACCENT3, alpha=0.6, width=0.6)
    ax4.axhline(0, color=GRID_COL, lw=0.8)
    ax4.set_ylabel('MACD', color=MUTED, fontsize=8)
    leg4 = ax4.legend(loc='upper left', fontsize=7, fancybox=True,
                      framealpha=0.3, labelcolor=TEXT_COL)
    leg4.get_frame().set_facecolor(CARD_BG)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b\'%y'))
    plt.setp(ax4.get_xticklabels(), rotation=30, ha='right', fontsize=7)

    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f'{symbol}_dashboard.png')
    if save:
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  2. Correlation heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(returns: pd.DataFrame, save: bool = True) -> str:
    corr = returns.corr()
    # Shorten labels
    short = {c: c[:6] for c in corr.columns}
    corr  = corr.rename(index=short, columns=short)

    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), max(9, n * 0.4)),
                            facecolor=DARK_BG)
    _apply_dark_style(fig, ax)

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(10, 140, s=80, l=45, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
                ax=ax, square=True, linewidths=0.3, linecolor=DARK_BG,
                annot=n <= 20, fmt='.1f', annot_kws={'size': 6},
                cbar_kws={'shrink': 0.7})

    ax.set_title('Return Correlation Matrix — NIFTY-50 Stocks',
                 color=TEXT_COL, fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(labelsize=7, colors=MUTED)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'correlation_heatmap.png')
    if save:
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  3. Portfolio pie chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_portfolio_allocation(portfolio_result: dict, save: bool = True) -> str:
    weights  = portfolio_result['Weights']
    profile  = portfolio_result['Profile']
    sectors  = portfolio_result.get('Sector_Alloc', {})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), facecolor=DARK_BG)
    _apply_dark_style(fig, [ax1, ax2])

    palette = [ACCENT1, ACCENT2, ACCENT4, GOLD, ACCENT3,
               '#f0883e', '#79c0ff', '#56d364', '#ff7b72', '#cae8ff']

    # Stock allocation
    syms = list(weights.keys())[:12]
    wts  = [weights[s] * 100 for s in syms]
    if len(weights) > 12:
        wts.append(sum(list(weights.values())[12:]) * 100)
        syms.append('Others')

    wedges, texts, autotexts = ax1.pie(
        wts, labels=syms,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        colors=palette[:len(syms)],
        pctdistance=0.75, startangle=140,
        wedgeprops={'edgecolor': DARK_BG, 'linewidth': 1.5},
        textprops={'color': TEXT_COL, 'fontsize': 8}
    )
    for at in autotexts:
        at.set_color(DARK_BG)
        at.set_fontsize(7)
        at.set_fontweight('bold')
    ax1.set_title(f'{profile} Portfolio — Stock Allocation',
                  color=TEXT_COL, fontsize=12, fontweight='bold')

    # Sector breakdown
    if sectors:
        sec_names = list(sectors.keys())
        sec_vals  = [sectors[s] for s in sec_names]
        wedges2, texts2, at2 = ax2.pie(
            sec_vals, labels=sec_names,
            autopct='%1.1f%%',
            colors=palette[:len(sec_names)],
            pctdistance=0.75, startangle=140,
            wedgeprops={'edgecolor': DARK_BG, 'linewidth': 1.5},
            textprops={'color': TEXT_COL, 'fontsize': 8}
        )
        for a in at2:
            a.set_color(DARK_BG)
            a.set_fontsize(7)
            a.set_fontweight('bold')
        ax2.set_title('Sector Breakdown', color=TEXT_COL, fontsize=12, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f'{profile.lower()}_allocation.png')
    if save:
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  4. Backtest performance chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest(bt_results: dict, labels: list = None, save: bool = True) -> str:
    """
    bt_results: {label: backtest_df} where backtest_df has Portfolio_Value and Drawdown columns
    """
    colors = [ACCENT1, ACCENT2, ACCENT4, GOLD, ACCENT3]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9),
                                    facecolor=DARK_BG, sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
    _apply_dark_style(fig, [ax1, ax2])

    for i, (lbl, df) in enumerate(bt_results.items()):
        c = colors[i % len(colors)]
        ax1.plot(df.index, df['Portfolio_Value'], color=c, lw=1.8, label=lbl, alpha=0.95)
        ax2.fill_between(df.index, df['Drawdown'] * 100, 0,
                         color=c, alpha=0.35, label=lbl)

    ax1.set_title('Portfolio Backtest — Cumulative Value (Base ₹100)',
                  color=TEXT_COL, fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('Portfolio Value (₹)', color=MUTED)
    leg = ax1.legend(fontsize=9, fancybox=True, framealpha=0.4, labelcolor=TEXT_COL)
    leg.get_frame().set_facecolor(CARD_BG)

    ax2.set_ylabel('Drawdown (%)', color=MUTED)
    ax2.set_xlabel('Date', color=MUTED)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'backtest_performance.png')
    if save:
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  5. Efficient Frontier
# ─────────────────────────────────────────────────────────────────────────────

def plot_efficient_frontier(frontier_df: pd.DataFrame,
                             portfolios: dict = None, save: bool = True) -> str:
    """
    frontier_df: columns [Return, Volatility, Sharpe]
    portfolios: {label: {Ann_Return, Ann_Volatility}}  optional overlay points
    """
    fig, ax = plt.subplots(figsize=(11, 7), facecolor=DARK_BG)
    _apply_dark_style(fig, ax)

    sc = ax.scatter(frontier_df['Volatility'], frontier_df['Return'],
                    c=frontier_df['Sharpe'], cmap='viridis',
                    s=25, alpha=0.8, zorder=3)
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('Sharpe Ratio', color=MUTED, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)

    # Overlay portfolio points
    if portfolios:
        markers = ['*', 'D', '^', 'P']
        palette = [ACCENT2, ACCENT1, ACCENT3, GOLD]
        for j, (lbl, pt) in enumerate(portfolios.items()):
            ax.scatter(pt['Ann_Volatility'], pt['Ann_Return'],
                       marker=markers[j % 4], s=220,
                       color=palette[j % 4], zorder=5,
                       edgecolors='white', linewidths=0.8, label=lbl)
            ax.annotate(lbl, (pt['Ann_Volatility'], pt['Ann_Return']),
                        xytext=(6, 4), textcoords='offset points',
                        color=TEXT_COL, fontsize=8, fontweight='bold')

    # Risk-free line
    if not frontier_df.empty:
        max_ret = frontier_df['Return'].max()
        ax.axhline(6, color=MUTED, linestyle=':', lw=1, alpha=0.5, label='Risk-Free (6%)')

    ax.set_xlabel('Annual Volatility (%)', color=MUTED)
    ax.set_ylabel('Annual Return (%)', color=MUTED)
    ax.set_title('Efficient Frontier — NIFTY-50 Portfolios',
                 color=TEXT_COL, fontsize=13, fontweight='bold', pad=10)
    leg = ax.legend(fontsize=9, fancybox=True, framealpha=0.4, labelcolor=TEXT_COL)
    leg.get_frame().set_facecolor(CARD_BG)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'efficient_frontier.png')
    if save:
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  6. Risk profile comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_comparison(risk_df: pd.DataFrame, metric: str = 'Sharpe_Ratio',
                          top_n: int = 20, save: bool = True) -> str:
    df = risk_df.nlargest(top_n, metric).sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.38)), facecolor=DARK_BG)
    _apply_dark_style(fig, ax)

    colors = [ACCENT2 if v > 0 else ACCENT3 for v in df[metric]]
    bars = ax.barh(df['Symbol'], df[metric], color=colors, alpha=0.85, edgecolor=DARK_BG)

    for bar, val in zip(bars, df[metric]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', ha='left', color=TEXT_COL, fontsize=8)

    ax.set_xlabel(metric.replace('_', ' '), color=MUTED)
    ax.set_title(f'Top {top_n} Stocks — {metric.replace("_", " ")}',
                 color=TEXT_COL, fontsize=12, fontweight='bold', pad=8)
    ax.axvline(0, color=MUTED, lw=0.8)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f'risk_{metric.lower()}.png')
    if save:
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  7. Sector performance heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_sector_performance(risk_df: pd.DataFrame, save: bool = True) -> str:
    metrics = ['Ann_Return', 'Ann_Volatility', 'Sharpe_Ratio', 'Max_Drawdown']
    avail   = [m for m in metrics if m in risk_df.columns]
    sector  = risk_df.groupby('Sector')[avail].mean().round(2)

    fig, ax = plt.subplots(figsize=(12, max(5, len(sector) * 0.55)), facecolor=DARK_BG)
    _apply_dark_style(fig, ax)

    norm = sector.copy()
    for col in norm.columns:
        rng = norm[col].max() - norm[col].min()
        if rng > 0:
            norm[col] = (norm[col] - norm[col].min()) / rng

    cmap = sns.diverging_palette(10, 145, s=80, l=45, as_cmap=True)
    sns.heatmap(norm, ax=ax, cmap=cmap, annot=sector, fmt='.1f',
                linewidths=0.5, linecolor=DARK_BG,
                cbar_kws={'shrink': 0.6},
                annot_kws={'size': 9, 'color': TEXT_COL})

    ax.set_title('Sector-Level Performance Summary',
                 color=TEXT_COL, fontsize=12, fontweight='bold', pad=8)
    ax.tick_params(labelsize=9, colors=TEXT_COL)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, 'sector_performance.png')
    if save:
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  8. Feature importance bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(importances: dict, symbol: str,
                             top_n: int = 15, save: bool = True) -> str:
    top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    feats, vals = zip(*top)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=DARK_BG)
    _apply_dark_style(fig, ax)

    colors = [ACCENT1 if v > np.mean(vals) else ACCENT4 for v in vals]
    bars = ax.barh(list(feats)[::-1], list(vals)[::-1],
                   color=colors[::-1], alpha=0.85, edgecolor=DARK_BG)

    ax.set_xlabel('Feature Importance', color=MUTED)
    ax.set_title(f'[{symbol}]  Top {top_n} Features — Random Forest',
                 color=TEXT_COL, fontsize=12, fontweight='bold', pad=8)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f'{symbol}_feature_importance.png')
    if save:
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
#  9. Anomaly plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_anomalies(df: pd.DataFrame, anomalies_df: pd.DataFrame,
                   symbol: str, save: bool = True) -> str:
    df = df.sort_values('Date')
    dates = pd.to_datetime(df['Date'])

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK_BG)
    _apply_dark_style(fig, ax)

    ax.plot(dates, df['Close'], color=ACCENT1, lw=1.4, alpha=0.9, label='Close Price')

    if not anomalies_df.empty:
        adf = anomalies_df.copy()
        adf['Date'] = pd.to_datetime(adf['Date'])

        crash = adf[adf['Type'] == 'Return Crash']
        spike = adf[adf['Type'] == 'Return Spike']
        vspike = adf[adf['Type'] == 'Volume Spike']

        for row in crash.itertuples():
            ax.axvline(row.Date, color=ACCENT3, alpha=0.4, lw=1)
        for row in spike.itertuples():
            ax.axvline(row.Date, color=ACCENT2, alpha=0.4, lw=1)

        if not crash.empty:
            ax.scatter(crash['Date'], crash['Close'], color=ACCENT3, s=60,
                       zorder=5, label=f'Crash ({len(crash)})', marker='v')
        if not spike.empty:
            ax.scatter(spike['Date'], spike['Close'], color=ACCENT2, s=60,
                       zorder=5, label=f'Spike ({len(spike)})', marker='^')

    ax.set_title(f'[{symbol}]  Price & Market Anomalies',
                 color=TEXT_COL, fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('Price (₹)', color=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)
    leg = ax.legend(fontsize=9, fancybox=True, framealpha=0.4, labelcolor=TEXT_COL)
    leg.get_frame().set_facecolor(CARD_BG)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f'{symbol}_anomalies.png')
    if save:
        plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    return path


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_stock, add_technical_indicators
    df = load_stock('TCS')
    df = add_technical_indicators(df)
    p = plot_stock_dashboard(df, 'TCS', 'Tata Consultancy Services')
    print(f"Saved: {p}")
