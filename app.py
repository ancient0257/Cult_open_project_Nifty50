"""
NIFTY-50 Investment Intelligence Platform
Main Streamlit Application
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys, warnings, pickle
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader   import (load_stock, add_technical_indicators,
                            build_returns_matrix, get_available_symbols,
                            SECTOR_MAP, DISPLAY_NAMES)
from risk_module   import (compute_stock_risk, compute_all_risk_profiles,
                            detect_anomalies, portfolio_risk, rolling_portfolio_metrics)
from portfolio_module import (build_conservative_portfolio, build_balanced_portfolio,
                               build_aggressive_portfolio, build_equal_weight_portfolio,
                               backtest_portfolio, efficient_frontier)
from predictor     import (train_stock_model, predict_latest, prepare_features)
from explainer     import (technical_signal_summary, explain_prediction,
                            explain_portfolio)
from visualizer    import (plot_stock_dashboard, plot_correlation_heatmap,
                            plot_portfolio_allocation, plot_backtest,
                            plot_efficient_frontier, plot_risk_comparison,
                            plot_sector_performance, plot_feature_importance,
                            plot_anomalies, OUTPUTS_DIR)

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #0d1117; color: #e6edf3; }
  .stSidebar { background: #161b22; }
  .metric-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 14px 18px; text-align: center;
  }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: #58a6ff; }
  .metric-label { font-size: 0.78rem; color: #8b949e; margin-top: 4px; }
  .green  { color: #3fb950 !important; }
  .red    { color: #f85149 !important; }
  .gold   { color: #e3b341 !important; }
  .purple { color: #d2a8ff !important; }
  .signal-badge {
    display:inline-block; padding:4px 12px; border-radius:20px;
    font-weight:700; font-size:0.85rem;
  }
  .badge-buy  { background:#1a4a2e; color:#3fb950; }
  .badge-sell { background:#4a1a1a; color:#f85149; }
  .badge-neutral { background:#2a2a1a; color:#e3b341; }
  h1,h2,h3 { color: #e6edf3; }
  .stTabs [data-baseweb="tab"] { color: #8b949e; font-weight:600; }
  .stTabs [aria-selected="true"] { color: #58a6ff; border-bottom:2px solid #58a6ff; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Cached data loading
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_all_risk_profiles():
    symbols = get_available_symbols()
    dfs = []
    for sym in symbols:
        try:
            df = load_stock(sym)
            df = add_technical_indicators(df)
            dfs.append(df)
        except:
            pass
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    return compute_all_risk_profiles(all_df)

@st.cache_data(show_spinner=False)
def get_returns_matrix():
    prices, returns = build_returns_matrix()
    returns_clean = returns.dropna(axis=1, thresh=int(len(returns) * 0.7)).fillna(0)
    return prices, returns_clean

@st.cache_data(show_spinner=False)
def get_stock_data(symbol):
    df = load_stock(symbol)
    df = add_technical_indicators(df)
    return df

@st.cache_data(show_spinner=False)
def get_all_portfolios():
    prices, returns = get_returns_matrix()
    risk_df = get_all_risk_profiles()
    conservative = build_conservative_portfolio(returns.copy(), risk_df)
    balanced     = build_balanced_portfolio(returns.copy(), risk_df)
    aggressive   = build_aggressive_portfolio(returns.copy(), risk_df)
    equal        = build_equal_weight_portfolio(returns.copy(), risk_df)
    return {'Conservative': conservative, 'Balanced': balanced,
            'Aggressive': aggressive, 'Equal Weight': equal}

@st.cache_data(show_spinner=False)
def get_efficient_frontier():
    _, returns = get_returns_matrix()
    return efficient_frontier(returns, n_points=60)

@st.cache_data(show_spinner=False)
def train_model_cached(symbol):
    df = get_stock_data(symbol)
    return train_stock_model(df, symbol)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def color_metric(val, good_if='high', threshold=0):
    if good_if == 'high':
        return 'green' if val > threshold else 'red'
    else:
        return 'green' if val < threshold else 'red'

def metric_card(label, value, suffix='', color='blue'):
    color_map = {'blue':'#58a6ff','green':'#3fb950','red':'#f85149',
                 'gold':'#e3b341','purple':'#d2a8ff'}
    c = color_map.get(color, '#58a6ff')
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-value" style="color:{c}">{value}{suffix}</div>
      <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

def signal_badge(signal):
    if 'BUY' in signal:
        cls = 'badge-buy'
    elif 'SELL' in signal:
        cls = 'badge-sell'
    else:
        cls = 'badge-neutral'
    return f'<span class="signal-badge {cls}">{signal}</span>'


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.image("https://img.shields.io/badge/NIFTY--50-Intelligence-blue?style=for-the-badge", width=260)
st.sidebar.markdown("## 📊 Navigation")

page = st.sidebar.radio("", [
    "🏠 Dashboard Overview",
    "📈 Stock Analyser",
    "🤖 AI Predictor",
    "💼 Portfolio Builder",
    "⚠️ Risk Assessment",
    "🔍 Anomaly Detector",
    "🌐 Market Overview",
], label_visibility='collapsed')

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
start_date = st.sidebar.selectbox("Data Start Year", ['2010', '2012', '2015', '2018'], index=0)
chart_window = st.sidebar.selectbox("Chart Window", ['1 Year', '2 Years', '5 Years', 'All'], index=0)
window_days = {'1 Year': 252, '2 Years': 504, '5 Years': 1260, 'All': 9999}[chart_window]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#8b949e'>NIFTY-50 · Jan 2000 – Apr 2021<br>"
    "Data: NSE Historical · Model: Random Forest<br>"
    "⚠️ Not financial advice</small>",
    unsafe_allow_html=True
)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 1: Dashboard Overview
# ─────────────────────────────────────────────────────────────────────────────

if page == "🏠 Dashboard Overview":
    st.title("📈 NIFTY-50 Investment Intelligence Platform")
    st.markdown("<p style='color:#8b949e'>AI-powered analytics · Portfolio optimisation · Risk assessment</p>",
                unsafe_allow_html=True)

    with st.spinner("Loading market data..."):
        risk_df = get_all_risk_profiles()

    if risk_df.empty:
        st.error("Could not load risk profiles. Check data directory.")
        st.stop()

    # ── Top KPIs ─────────────────────────────────────────────────────────────
    st.markdown("### 📊 Market Overview — All NIFTY-50 Stocks")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Stocks Tracked", len(risk_df), color='blue')
    with c2: metric_card("Avg Annual Return", f"{risk_df['Ann_Return'].mean():.1f}", suffix="%", color='green')
    with c3: metric_card("Avg Volatility", f"{risk_df['Ann_Volatility'].mean():.1f}", suffix="%", color='gold')
    with c4: metric_card("Avg Sharpe Ratio", f"{risk_df['Sharpe_Ratio'].mean():.2f}", color='purple')
    with c5: metric_card("Avg Max Drawdown", f"{risk_df['Max_Drawdown'].mean():.1f}", suffix="%", color='red')

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("#### 🏆 Top 10 Stocks by Sharpe Ratio")
        top10 = risk_df.nlargest(10, 'Sharpe_Ratio')[
            ['Name', 'Sector', 'Ann_Return', 'Ann_Volatility', 'Sharpe_Ratio', 'Max_Drawdown']
        ].reset_index(drop=True)
        top10.index += 1
        st.dataframe(
            top10.style
            .background_gradient(subset=['Sharpe_Ratio'], cmap='Greens')
            .background_gradient(subset=['Ann_Return'],   cmap='Blues')
            .format({'Ann_Return': '{:.1f}%', 'Ann_Volatility': '{:.1f}%',
                     'Sharpe_Ratio': '{:.3f}', 'Max_Drawdown': '{:.1f}%'}),
            use_container_width=True, height=370
        )

    with col2:
        st.markdown("#### 🗂 Sector Risk Distribution")
        sec_summary = risk_df.groupby('Sector').agg(
            Count=('Symbol', 'count'),
            Avg_Return=('Ann_Return', 'mean'),
            Avg_Vol=('Ann_Volatility', 'mean'),
            Avg_Sharpe=('Sharpe_Ratio', 'mean')
        ).round(2).sort_values('Avg_Sharpe', ascending=False)
        st.dataframe(
            sec_summary.style
            .background_gradient(subset=['Avg_Sharpe'], cmap='RdYlGn')
            .format({'Avg_Return': '{:.1f}%', 'Avg_Vol': '{:.1f}%', 'Avg_Sharpe': '{:.2f}'}),
            use_container_width=True, height=370
        )

    # ── Sector performance chart ─────────────────────────────────────────────
    with st.spinner("Generating sector chart..."):
        sector_path = plot_sector_performance(risk_df)
    st.image(sector_path, use_container_width=True)

    # ── Sharpe bar chart ─────────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        sharpe_path = plot_risk_comparison(risk_df, 'Sharpe_Ratio', top_n=15)
        st.image(sharpe_path, use_container_width=True)
    with col4:
        ret_path = plot_risk_comparison(risk_df, 'Ann_Return', top_n=15)
        st.image(ret_path, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 2: Stock Analyser
# ─────────────────────────────────────────────────────────────────────────────

elif page == "📈 Stock Analyser":
    st.title("📈 Stock Analyser")

    symbols = get_available_symbols()
    display_options = {f"{DISPLAY_NAMES.get(s, s)} ({s})": s for s in symbols}
    sel = st.selectbox("Select Stock", list(display_options.keys()), index=symbols.index('RELIANCE'))
    symbol = display_options[sel]

    with st.spinner(f"Loading {symbol}..."):
        df = get_stock_data(symbol)

    name   = DISPLAY_NAMES.get(symbol, symbol)
    sector = SECTOR_MAP.get(symbol, 'Unknown')

    # KPIs
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    change = (latest['Close'] - prev['Close']) / prev['Close'] * 100

    st.markdown(f"### {name}  `{symbol}`  ·  <span style='color:#8b949e'>{sector}</span>",
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    chg_color = 'green' if change >= 0 else 'red'
    with c1: metric_card("Latest Close", f"₹{latest['Close']:.2f}", color='blue')
    with c2: metric_card("1-Day Change", f"{change:+.2f}", suffix="%", color=chg_color)
    with c3: metric_card("RSI (14)", f"{latest.get('RSI_14', 0):.1f}", color='purple')
    with c4: metric_card("20d Volatility", f"{latest.get('Volatility_20', 0)*100:.1f}", suffix="%", color='gold')
    with c5: metric_card("MACD", f"{latest.get('MACD', 0):.2f}",
                          color='green' if latest.get('MACD', 0) > 0 else 'red')

    st.markdown("<br>", unsafe_allow_html=True)

    # Technical Signal Summary
    sig = technical_signal_summary(df)
    st.markdown("#### 🔔 Technical Signal Summary")
    bcols = st.columns(len(sig['Signals']))
    sig_colors = {'Bullish':'green','Golden Cross':'green','Oversold':'green',
                  'Bearish':'red','Death Cross':'red','Overbought':'red',
                  'Near Upper':'red'}
    for i, (k, v) in enumerate(sig['Signals'].items()):
        with bcols[i]:
            c = sig_colors.get(v['value'], 'gold')
            metric_card(k.replace('_', ' '), v['value'], color=c)

    overall_color = 'green' if sig['Overall'] == 'BULLISH' else ('red' if sig['Overall'] == 'BEARISH' else 'gold')
    st.markdown(f"**Overall Signal:** <span style='color:{'#3fb950' if overall_color=='green' else '#f85149' if overall_color=='red' else '#e3b341'}; font-size:1.1rem; font-weight:700'>{sig['Overall']}</span> &nbsp;&nbsp; ({sig['Bullish']} bullish / {sig['Bearish']} bearish indicators)",
                unsafe_allow_html=True)

    # Chart
    with st.spinner("Rendering chart..."):
        chart_path = plot_stock_dashboard(df, symbol, name, last_n=window_days)
    st.image(chart_path, use_container_width=True)

    # Historical stats table
    with st.expander("📋 Historical Statistics"):
        risk = compute_stock_risk(df, symbol)
        risk_display = {
            'Annual Return (%)':    risk['Ann_Return'],
            'Annual Volatility (%)':risk['Ann_Volatility'],
            'Sharpe Ratio':         risk['Sharpe_Ratio'],
            'Sortino Ratio':        risk['Sortino_Ratio'],
            'Max Drawdown (%)':     risk['Max_Drawdown'],
            'Calmar Ratio':         risk['Calmar_Ratio'],
            'VaR 95% (%)':          risk['VaR_95'],
            'CVaR 95% (%)':         risk['CVaR_95'],
            'Skewness':             risk['Skewness'],
            'Kurtosis':             risk['Kurtosis'],
        }
        rd = pd.DataFrame.from_dict(risk_display, orient='index', columns=['Value'])
        st.dataframe(rd, use_container_width=True)

    # Raw data
    with st.expander("📊 Raw Data (Last 50 rows)"):
        show_cols = ['Date','Open','High','Low','Close','Volume',
                     'MA_20','MA_50','RSI_14','MACD','BB_Upper','BB_Lower',
                     'Volatility_20','Daily_Return']
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols].tail(50).sort_values('Date', ascending=False)
                     .reset_index(drop=True), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 3: AI Predictor
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🤖 AI Predictor":
    st.title("🤖 AI Stock Predictor")
    st.markdown("<p style='color:#8b949e'>Random Forest classifier for directional prediction + return forecasting. Walk-forward evaluated.</p>",
                unsafe_allow_html=True)

    symbols = get_available_symbols()
    display_options = {f"{DISPLAY_NAMES.get(s, s)} ({s})": s for s in symbols}
    sel    = st.selectbox("Select Stock", list(display_options.keys()), index=symbols.index('TCS'))
    symbol = display_options[sel]

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_model = st.button("🚀 Train & Predict", type='primary', use_container_width=True)

    if run_model or f'model_{symbol}' in st.session_state:
        with st.spinner(f"Training model for {symbol}... (~10 seconds)"):
            result = train_model_cached(symbol)
            st.session_state[f'model_{symbol}'] = result

        result  = st.session_state[f'model_{symbol}']
        metrics = result['metrics']
        df      = get_stock_data(symbol)
        pred    = predict_latest(df, result)

        # Prediction card
        st.markdown("---")
        st.markdown("### 🎯 Latest Prediction")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Signal", pred.get('Signal', '-'),
                        color='green' if 'BUY' in pred.get('Signal','') else
                              'red'   if 'SELL' in pred.get('Signal','') else 'gold')
        with c2:
            metric_card("Direction", pred.get('Direction', '-'),
                        color='green' if pred.get('Direction') == 'UP' else 'red')
        with c3:
            metric_card("Prob (Up)", f"{pred.get('Direction_Prob', 50):.1f}", suffix="%",
                        color='green' if pred.get('Direction_Prob', 50) > 55 else
                              'red'   if pred.get('Direction_Prob', 50) < 45 else 'gold')
        with c4:
            ret5 = pred.get('Pred_Return_5d', 0)
            metric_card("5-Day Return Forecast", f"{ret5:+.2f}", suffix="%",
                        color='green' if ret5 > 0 else 'red')

        # XAI explanation
        latest_row = df.sort_values('Date').iloc[-1]
        imp = result.get('importances', {})
        if imp:
            from explainer import explain_prediction
            explanation = explain_prediction(latest_row, pred, imp)
            st.markdown("---")
            st.markdown("### 💡 AI Explanation (XAI)")
            st.info(explanation['Summary'].replace('**', ''))

            st.markdown("#### Top Contributing Indicators")
            exp_df = pd.DataFrame(explanation['Reasons'])[
                ['Feature', 'Description', 'Value', 'Interpretation', 'Signal', 'Importance_Pct']
            ]
            st.dataframe(
                exp_df.style.applymap(
                    lambda v: 'color: #3fb950' if v == 'BULLISH' else
                              'color: #f85149' if v == 'BEARISH' else '',
                    subset=['Signal']
                ),
                use_container_width=True, hide_index=True
            )

        # Model performance
        st.markdown("---")
        st.markdown("### 📊 Model Performance Metrics")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: metric_card("Accuracy",  f"{metrics['Accuracy']:.1f}",  suffix="%", color='blue')
        with c2: metric_card("Precision", f"{metrics['Precision']:.1f}", suffix="%", color='purple')
        with c3: metric_card("Recall",    f"{metrics['Recall']:.1f}",    suffix="%", color='gold')
        with c4: metric_card("F1 Score",  f"{metrics['F1']:.1f}",        suffix="%", color='green')
        with c5: metric_card("MAE (5d)",  f"{metrics['MAE_5d']:.2f}",    suffix="%", color='gold')
        with c6: metric_card("R² (5d)",   f"{metrics['R2_5d']:.3f}",     color='blue')

        # Feature importance chart
        if imp:
            st.markdown("### 🔍 Feature Importances")
            fi_path = plot_feature_importance(imp, symbol, top_n=15)
            st.image(fi_path, use_container_width=True)

        # Test-set predictions
        with st.expander("📋 Test-Set Predictions (Last 50)"):
            tp = result['test_preds'].tail(50).sort_values('Date', ascending=False)
            tp['Date'] = tp['Date'].astype(str).str[:10]
            tp['Correct'] = (tp['Actual_Dir'] == tp['Pred_Dir']).map({True:'✅', False:'❌'})
            st.dataframe(tp[['Date','Actual_Dir','Pred_Dir','Correct',
                              'Dir_Prob_Up','Actual_Ret5d','Pred_Ret5d']]
                         .rename(columns={'Dir_Prob_Up':'P(UP)',
                                          'Actual_Ret5d':'Actual 5d%',
                                          'Pred_Ret5d':'Pred 5d%'})
                         .reset_index(drop=True),
                         use_container_width=True)
    else:
        st.info("👆 Select a stock and click **Train & Predict** to run the AI model.")


# ─────────────────────────────────────────────────────────────────────────────
#  Page 4: Portfolio Builder
# ─────────────────────────────────────────────────────────────────────────────

elif page == "💼 Portfolio Builder":
    st.title("💼 Portfolio Builder")
    st.markdown("<p style='color:#8b949e'>Markowitz mean-variance optimisation across three investor profiles.</p>",
                unsafe_allow_html=True)

    with st.spinner("Optimising portfolios..."):
        portfolios  = get_all_portfolios()
        risk_df     = get_all_risk_profiles()
        prices, _   = get_returns_matrix()
        frontier_df = get_efficient_frontier()

    profile_tabs = st.tabs(["🛡 Conservative", "⚖️ Balanced", "🚀 Aggressive", "🎲 Equal Weight", "📊 Comparison"])

    def render_portfolio_tab(pf_result, risk_df):
        profile = pf_result['Profile']
        metrics = pf_result['Metrics']
        weights = pf_result['Weights']

        # KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: metric_card("Annual Return",    f"{metrics['Ann_Return']:.1f}", suffix="%",
                              color='green' if metrics['Ann_Return'] > 10 else 'gold')
        with c2: metric_card("Annual Volatility",f"{metrics['Ann_Volatility']:.1f}", suffix="%", color='gold')
        with c3: metric_card("Sharpe Ratio",     f"{metrics['Sharpe_Ratio']:.3f}", color='blue')
        with c4: metric_card("Max Drawdown",     f"{metrics['Max_Drawdown']:.1f}", suffix="%", color='red')
        with c5: metric_card("Holdings",         pf_result['N_Holdings'], color='purple')

        st.markdown("<br>", unsafe_allow_html=True)

        # XAI narrative
        exp = explain_portfolio(pf_result, risk_df)
        st.info(exp['Portfolio_Narrative'].replace('**', ''))

        col1, col2 = st.columns([1, 1.2])
        with col1:
            alloc_path = plot_portfolio_allocation(pf_result)
            st.image(alloc_path, use_container_width=True)

        with col2:
            st.markdown("#### 📋 Holdings & Justification")
            hold_df = pd.DataFrame(exp['Holdings'])
            if not hold_df.empty:
                cols_show = [c for c in ['Symbol','Weight_Pct','Sector','Sharpe',
                                          'Ann_Return','Ann_Vol','Justification'] if c in hold_df.columns]
                st.dataframe(
                    hold_df[cols_show].style
                    .background_gradient(subset=['Weight_Pct'], cmap='Blues')
                    .format({c: '{:.1f}' for c in ['Weight_Pct','Ann_Return','Ann_Vol','Sharpe'] if c in hold_df.columns}),
                    use_container_width=True, height=380
                )

        # Backtest
        st.markdown("#### 📈 Backtest Performance")
        bt_df = backtest_portfolio(weights, prices)
        bt_label = profile
        with st.spinner("Running backtest..."):
            bt_path = plot_backtest({bt_label: bt_df})
        st.image(bt_path, use_container_width=True)

        final_ret = bt_df['Cumulative_Return'].iloc[-1] * 100
        st.markdown(f"**Total Return (backtested):** <span style='color:#3fb950;font-weight:700'>{final_ret:+.1f}%</span>",
                    unsafe_allow_html=True)

    with profile_tabs[0]: render_portfolio_tab(portfolios['Conservative'], risk_df)
    with profile_tabs[1]: render_portfolio_tab(portfolios['Balanced'],     risk_df)
    with profile_tabs[2]: render_portfolio_tab(portfolios['Aggressive'],   risk_df)
    with profile_tabs[3]: render_portfolio_tab(portfolios['Equal Weight'], risk_df)

    with profile_tabs[4]:
        st.markdown("### 📊 Portfolio Comparison")

        comp_rows = []
        for lbl, pf in portfolios.items():
            row = {'Profile': lbl}
            row.update(pf['Metrics'])
            comp_rows.append(row)
        comp_df = pd.DataFrame(comp_rows)
        st.dataframe(
            comp_df.style
            .highlight_max(subset=['Ann_Return','Sharpe_Ratio','Sortino_Ratio','Calmar_Ratio'],
                           color='#1a4a2e')
            .highlight_min(subset=['Ann_Volatility','Max_Drawdown'], color='#1a4a2e')
            .format({c: '{:.2f}' for c in comp_df.select_dtypes('number').columns}),
            use_container_width=True, hide_index=True
        )

        # Efficient frontier
        st.markdown("### 🎯 Efficient Frontier")
        overlay = {}
        for lbl, pf in portfolios.items():
            overlay[lbl] = {
                'Ann_Return':     pf['Metrics']['Ann_Return'],
                'Ann_Volatility': pf['Metrics']['Ann_Volatility'],
            }
        with st.spinner("Plotting efficient frontier..."):
            ef_path = plot_efficient_frontier(frontier_df, overlay)
        st.image(ef_path, use_container_width=True)

        # Combined backtest
        bt_results = {}
        for lbl, pf in portfolios.items():
            bt_results[lbl] = backtest_portfolio(pf['Weights'], prices)
        combined_bt_path = plot_backtest(bt_results)
        st.image(combined_bt_path, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Page 5: Risk Assessment
# ─────────────────────────────────────────────────────────────────────────────

elif page == "⚠️ Risk Assessment":
    st.title("⚠️ Risk Assessment")
    st.markdown("<p style='color:#8b949e'>Volatility, Sharpe, Sortino, Max Drawdown, VaR, CVaR across all NIFTY-50 stocks.</p>",
                unsafe_allow_html=True)

    with st.spinner("Computing risk profiles..."):
        risk_df = get_all_risk_profiles()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        sectors = ['All'] + sorted(risk_df['Sector'].unique().tolist())
        sel_sec = st.selectbox("Filter by Sector", sectors)
    with col2:
        risk_tiers = ['All', 'Low', 'Medium', 'High']
        sel_tier = st.selectbox("Filter by Risk Tier", risk_tiers)
    with col3:
        sort_by = st.selectbox("Sort By", ['Sharpe_Ratio','Ann_Return','Ann_Volatility',
                                            'Sortino_Ratio','Max_Drawdown'])

    filtered = risk_df.copy()
    if sel_sec != 'All':
        filtered = filtered[filtered['Sector'] == sel_sec]
    if sel_tier != 'All':
        filtered = filtered[filtered['Risk_Tier'] == sel_tier]
    filtered = filtered.sort_values(sort_by, ascending=(sort_by in ['Ann_Volatility','Max_Drawdown']))

    # Full table
    display_cols = ['Name','Symbol','Sector','Risk_Tier','Ann_Return','Ann_Volatility',
                    'Sharpe_Ratio','Sortino_Ratio','Max_Drawdown','Calmar_Ratio',
                    'VaR_95','CVaR_95','Rolling_Vol_60d']
    st.dataframe(
        filtered[display_cols].reset_index(drop=True).style
        .background_gradient(subset=['Sharpe_Ratio'], cmap='Greens')
        .background_gradient(subset=['Ann_Volatility','Max_Drawdown'], cmap='Reds_r')
        .format({'Ann_Return':'{:.1f}%','Ann_Volatility':'{:.1f}%',
                 'Sharpe_Ratio':'{:.3f}','Sortino_Ratio':'{:.3f}',
                 'Max_Drawdown':'{:.1f}%','Calmar_Ratio':'{:.3f}',
                 'VaR_95':'{:.2f}%','CVaR_95':'{:.2f}%','Rolling_Vol_60d':'{:.1f}%'}),
        use_container_width=True, height=450
    )

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        path1 = plot_risk_comparison(filtered, 'Sharpe_Ratio', top_n=min(20, len(filtered)))
        st.image(path1, use_container_width=True)
    with col2:
        path2 = plot_risk_comparison(filtered, 'Max_Drawdown', top_n=min(20, len(filtered)))
        st.image(path2, use_container_width=True)

    # Individual deep-dive
    st.markdown("---")
    st.markdown("### 🔬 Deep-Dive: Individual Stock Risk")
    symbols = get_available_symbols()
    display_options = {f"{DISPLAY_NAMES.get(s,s)} ({s})": s for s in symbols}
    sel    = st.selectbox("Select Stock for Risk Deep-Dive", list(display_options.keys()),
                           index=symbols.index('HDFCBANK'))
    symbol = display_options[sel]

    df = get_stock_data(symbol)
    risk = compute_stock_risk(df, symbol)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: metric_card("Annual Return",   f"{risk['Ann_Return']:.1f}", suffix="%", color='green')
    with c2: metric_card("Volatility",      f"{risk['Ann_Volatility']:.1f}", suffix="%", color='gold')
    with c3: metric_card("Sharpe",          f"{risk['Sharpe_Ratio']:.3f}", color='blue')
    with c4: metric_card("Sortino",         f"{risk['Sortino_Ratio']:.3f}", color='purple')
    with c5: metric_card("Max Drawdown",    f"{risk['Max_Drawdown']:.1f}", suffix="%", color='red')
    with c6: metric_card("VaR 95%",         f"{risk['VaR_95']:.2f}", suffix="%", color='red')


# ─────────────────────────────────────────────────────────────────────────────
#  Page 6: Anomaly Detector
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Anomaly Detector":
    st.title("🔍 Market Anomaly Detector")
    st.markdown("<p style='color:#8b949e'>Detects return spikes, crashes, and volume anomalies using z-score analysis.</p>",
                unsafe_allow_html=True)

    symbols = get_available_symbols()
    display_options = {f"{DISPLAY_NAMES.get(s,s)} ({s})": s for s in symbols}

    col1, col2 = st.columns([2, 1])
    with col1:
        sel    = st.selectbox("Select Stock", list(display_options.keys()),
                               index=symbols.index('TATAMOTORS'))
        symbol = display_options[sel]
    with col2:
        z_thresh = st.slider("Z-Score Threshold", 2.0, 5.0, 3.0, 0.5)

    df = get_stock_data(symbol)
    anomalies = detect_anomalies(df, z_threshold=z_thresh)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Total Anomalies", len(anomalies), color='red')
    crashes = anomalies[anomalies['Type']=='Return Crash'] if not anomalies.empty else pd.DataFrame()
    spikes  = anomalies[anomalies['Type']=='Return Spike'] if not anomalies.empty else pd.DataFrame()
    vspikes = anomalies[anomalies['Type']=='Volume Spike'] if not anomalies.empty else pd.DataFrame()
    with c2: metric_card("Return Crashes", len(crashes), color='red')
    with c3: metric_card("Return Spikes",  len(spikes),  color='green')
    with c4: metric_card("Volume Spikes",  len(vspikes), color='gold')

    with st.spinner("Plotting anomalies..."):
        path = plot_anomalies(df, anomalies, symbol)
    st.image(path, use_container_width=True)

    if not anomalies.empty:
        st.markdown("#### 📋 Anomaly Events Table")
        show_a = anomalies.copy()
        show_a['Date'] = show_a['Date'].astype(str).str[:10]
        st.dataframe(
            show_a.style
            .applymap(lambda v: 'color:#f85149' if v == 'Return Crash'
                      else 'color:#3fb950' if v == 'Return Spike'
                      else 'color:#e3b341', subset=['Type'])
            .applymap(lambda v: 'color:#f85149' if v == 'High' else '', subset=['Severity']),
            use_container_width=True, hide_index=True
        )
    else:
        st.success("No anomalies detected at this threshold.")


# ─────────────────────────────────────────────────────────────────────────────
#  Page 7: Market Overview
# ─────────────────────────────────────────────────────────────────────────────

elif page == "🌐 Market Overview":
    st.title("🌐 Market Overview")
    st.markdown("<p style='color:#8b949e'>Cross-stock correlations and index-level analytics.</p>",
                unsafe_allow_html=True)

    with st.spinner("Computing correlations..."):
        prices, returns = get_returns_matrix()

    # Correlation heatmap
    st.markdown("### 🔗 Return Correlation Matrix")
    corr_path = plot_correlation_heatmap(returns)
    st.image(corr_path, use_container_width=True)

    # Cumulative returns comparison
    st.markdown("### 📈 Cumulative Return Comparison")
    risk_df = get_all_risk_profiles()
    top_syms = risk_df.nlargest(10, 'Sharpe_Ratio')['Symbol'].tolist()

    sel_syms = st.multiselect("Select Stocks to Compare", get_available_symbols(),
                               default=top_syms[:5])
    if sel_syms:
        cum_returns = {}
        for sym in sel_syms:
            if sym in prices.columns:
                p = prices[sym].dropna()
                cum_returns[sym] = (p / p.iloc[0] - 1) * 100

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(14, 6), facecolor='#0d1117')
        ax.set_facecolor('#161b22')
        palette = ['#58a6ff','#3fb950','#f85149','#e3b341','#d2a8ff',
                   '#f0883e','#79c0ff','#56d364','#ff7b72','#cae8ff']
        for i, (sym, cr) in enumerate(cum_returns.items()):
            ax.plot(cr.index, cr.values, lw=1.6, color=palette[i % len(palette)],
                    label=DISPLAY_NAMES.get(sym, sym), alpha=0.9)
        ax.axhline(0, color='#30363d', lw=0.8, linestyle='--')
        ax.set_ylabel('Cumulative Return (%)', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(True, color='#30363d', alpha=0.5)
        leg = ax.legend(fontsize=9, fancybox=True, framealpha=0.4, labelcolor='#e6edf3')
        leg.get_frame().set_facecolor('#161b22')
        ax.set_title('Cumulative Returns (%)', color='#e6edf3', fontsize=12, fontweight='bold')
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig)
        buf.seek(0)
        st.image(buf, use_container_width=True)

    # Highest-correlated pairs
    st.markdown("### 🔗 Highest Correlated Pairs")
    corr = returns.corr()
    pairs = []
    for i, s1 in enumerate(corr.columns):
        for j, s2 in enumerate(corr.columns):
            if j > i:
                pairs.append({'Stock 1': s1, 'Stock 2': s2, 'Correlation': round(corr.iloc[i, j], 3)})
    pairs_df = pd.DataFrame(pairs)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 Positive Correlations**")
        st.dataframe(pairs_df.nlargest(10, 'Correlation').reset_index(drop=True),
                     use_container_width=True, hide_index=True)
    with col2:
        st.markdown("**Top 10 Negative / Lowest Correlations**")
        st.dataframe(pairs_df.nsmallest(10, 'Correlation').reset_index(drop=True),
                     use_container_width=True, hide_index=True)
