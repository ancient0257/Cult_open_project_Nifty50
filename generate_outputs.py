"""
Pre-generation script — run ONCE to generate all cached outputs.
Generates: risk profiles, charts, model results, portfolio results.
Saves to outputs/ and models/ directories.

Run: python3 generate_outputs.py
"""

import os, sys, warnings, json
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import (load_stock, add_technical_indicators,
                          build_returns_matrix, get_available_symbols,
                          DISPLAY_NAMES, SECTOR_MAP)
from risk_module import (compute_all_risk_profiles, detect_anomalies)
from portfolio_module import (build_conservative_portfolio, build_balanced_portfolio,
                               build_aggressive_portfolio, build_equal_weight_portfolio,
                               backtest_portfolio, efficient_frontier)
from predictor import train_stock_model, save_model
from visualizer import (plot_stock_dashboard, plot_correlation_heatmap,
                         plot_portfolio_allocation, plot_backtest,
                         plot_efficient_frontier, plot_risk_comparison,
                         plot_sector_performance, plot_feature_importance,
                         plot_anomalies, OUTPUTS_DIR)

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 1: Loading all stocks + technical indicators")
# ─────────────────────────────────────────────────────────────────────────────

symbols = get_available_symbols()
print(f"Loading {len(symbols)} stocks...")

all_dfs = []
for sym in symbols:
    try:
        df = load_stock(sym)
        df = add_technical_indicators(df)
        all_dfs.append(df)
        print(f"  ✓ {sym:15s}  {len(df)} rows")
    except Exception as e:
        print(f"  ✗ {sym}: {e}")

import pandas as pd
all_df = pd.concat(all_dfs, ignore_index=True)
print(f"\nTotal rows: {len(all_df):,}")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 2: Risk profiles for all stocks")
# ─────────────────────────────────────────────────────────────────────────────

risk_df = compute_all_risk_profiles(all_df)
out_path = os.path.join(OUTPUTS_DIR, 'risk_profiles.csv')
risk_df.to_csv(out_path, index=False)
print(f"Saved risk profiles → {out_path}")
print(risk_df[['Symbol','Ann_Return','Ann_Volatility','Sharpe_Ratio','Max_Drawdown']].head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 3: Returns matrix + correlation heatmap")
# ─────────────────────────────────────────────────────────────────────────────

prices, returns = build_returns_matrix()
returns_clean = returns.dropna(axis=1, thresh=int(len(returns) * 0.7)).fillna(0)

path = plot_correlation_heatmap(returns_clean)
print(f"Saved correlation heatmap → {path}")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 4: Portfolio optimisation (3 profiles + equal weight)")
# ─────────────────────────────────────────────────────────────────────────────

portfolios = {
    'Conservative': build_conservative_portfolio(returns_clean.copy(), risk_df),
    'Balanced':     build_balanced_portfolio(returns_clean.copy(), risk_df),
    'Aggressive':   build_aggressive_portfolio(returns_clean.copy(), risk_df),
    'Equal Weight': build_equal_weight_portfolio(returns_clean.copy(), risk_df),
}

for name, pf in portfolios.items():
    m = pf['Metrics']
    print(f"\n  [{name}]")
    print(f"    Holdings:     {pf['N_Holdings']}")
    print(f"    Ann Return:   {m['Ann_Return']:.2f}%")
    print(f"    Volatility:   {m['Ann_Volatility']:.2f}%")
    print(f"    Sharpe:       {m['Sharpe_Ratio']:.3f}")
    print(f"    Max Drawdown: {m['Max_Drawdown']:.2f}%")

    # Save allocation chart
    alloc_path = plot_portfolio_allocation(pf)
    print(f"    Chart → {alloc_path}")

# Save portfolio results
pf_out = {}
for name, pf in portfolios.items():
    pf_out[name] = {
        'Weights':     pf['Weights'],
        'Metrics':     pf['Metrics'],
        'Sector_Alloc': pf['Sector_Alloc'],
        'N_Holdings':  pf['N_Holdings'],
        'Profile':     pf['Profile'],
        'Description': pf['Description'],
    }
with open(os.path.join(OUTPUTS_DIR, 'portfolios.json'), 'w') as f:
    json.dump(pf_out, f, indent=2)
print(f"\nSaved portfolios.json")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 5: Backtest all portfolios")
# ─────────────────────────────────────────────────────────────────────────────

bt_results = {}
for name, pf in portfolios.items():
    try:
        bt = backtest_portfolio(pf['Weights'], prices)
        bt_results[name] = bt
        total_ret = bt['Cumulative_Return'].iloc[-1] * 100
        print(f"  {name}: Total Return = {total_ret:+.1f}%")
    except Exception as e:
        print(f"  {name}: backtest failed — {e}")

if bt_results:
    path = plot_backtest(bt_results)
    print(f"Saved backtest chart → {path}")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 6: Efficient Frontier")
# ─────────────────────────────────────────────────────────────────────────────

try:
    frontier_df = efficient_frontier(returns_clean, n_points=60)
    overlay = {name: {'Ann_Return': pf['Metrics']['Ann_Return'],
                       'Ann_Volatility': pf['Metrics']['Ann_Volatility']}
               for name, pf in portfolios.items()}
    path = plot_efficient_frontier(frontier_df, overlay)
    print(f"Saved efficient frontier → {path}")
except Exception as e:
    print(f"  Efficient frontier failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 7: Sector performance chart")
# ─────────────────────────────────────────────────────────────────────────────

path = plot_sector_performance(risk_df)
print(f"Saved sector chart → {path}")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 8: Risk comparison charts")
# ─────────────────────────────────────────────────────────────────────────────

for metric in ['Sharpe_Ratio', 'Ann_Return', 'Max_Drawdown', 'Ann_Volatility']:
    path = plot_risk_comparison(risk_df, metric, top_n=20)
    print(f"Saved {metric} chart → {path}")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 9: Train ML models for all 50 stocks")
# ─────────────────────────────────────────────────────────────────────────────

model_summary = []
for sym, gdf in all_df.groupby('Symbol'):
    try:
        result = train_stock_model(gdf.copy(), sym, model_type='rf')
        save_model(result, sym)
        m = result['metrics']
        model_summary.append(m)
        print(f"  [{sym:15s}] Acc={m['Accuracy']:.1f}%  Prec={m['Precision']:.1f}%  R2={m['R2_5d']:.3f}")

        # Feature importance chart
        if result.get('importances'):
            plot_feature_importance(result['importances'], sym, top_n=15)
    except Exception as e:
        print(f"  Warning [{sym}]: {e}")

model_df = pd.DataFrame(model_summary)
model_df.to_csv(os.path.join(OUTPUTS_DIR, 'model_summary.csv'), index=False)
print(f"\nSaved model summary → {OUTPUTS_DIR}/model_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 10: Stock dashboards for key stocks")
# ─────────────────────────────────────────────────────────────────────────────

key_stocks = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
              'KOTAKBANK', 'HINDUNILVR', 'ASIANPAINT', 'BAJFINANCE', 'MARUTI']
for sym in key_stocks:
    try:
        gdf = all_df[all_df['Symbol'] == sym]
        path = plot_stock_dashboard(gdf, sym, DISPLAY_NAMES.get(sym, sym))
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  {sym}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 11: Anomaly detection for all stocks")
# ─────────────────────────────────────────────────────────────────────────────

all_anomalies = []
for sym, gdf in all_df.groupby('Symbol'):
    try:
        adf = detect_anomalies(gdf)
        if not adf.empty:
            all_anomalies.append(adf)
    except:
        pass

if all_anomalies:
    combined_anomalies = pd.concat(all_anomalies, ignore_index=True)
    combined_anomalies.to_csv(os.path.join(OUTPUTS_DIR, 'all_anomalies.csv'), index=False)
    print(f"Total anomalies detected: {len(combined_anomalies)}")
    print(f"Saved → {OUTPUTS_DIR}/all_anomalies.csv")

# ─────────────────────────────────────────────────────────────────────────────
sep("DONE — All outputs generated!")
# ─────────────────────────────────────────────────────────────────────────────

out_files = os.listdir(OUTPUTS_DIR)
print(f"\n{len(out_files)} files in outputs/:")
for f in sorted(out_files):
    size = os.path.getsize(os.path.join(OUTPUTS_DIR, f))
    print(f"  {f:45s}  {size/1024:.1f} KB")

print("\n✅ Pre-generation complete. Run app with:")
print("   streamlit run app.py")
