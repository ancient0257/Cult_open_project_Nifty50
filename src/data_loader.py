"""
Data Loader & Feature Engineering
NIFTY-50 Investment Intelligence Platform
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
STOCKS_DIR = os.path.join(DATA_DIR, 'stocks')
INDEX_DIR  = os.path.join(DATA_DIR, 'index')

# Sector mapping for NIFTY-50 stocks
SECTOR_MAP = {
    'ADANIPORTS':  'Infrastructure',
    'ASIANPAINT':  'Consumer Goods',
    'AXISBANK':    'Banking',
    'BAJAJ-AUTO':  'Automobile',
    'BAJFINANCE':  'Financial Services',
    'BAJAJFINSV':  'Financial Services',
    'BPCL':        'Energy',
    'BHARTIARTL':  'Telecom',
    'BRITANNIA':   'Consumer Goods',
    'CIPLA':       'Pharmaceuticals',
    'COALINDIA':   'Energy',
    'DIVISLAB':    'Pharmaceuticals',
    'DRREDDY':     'Pharmaceuticals',
    'EICHERMOT':   'Automobile',
    'GRASIM':      'Cement & Construction',
    'HCLTECH':     'Information Technology',
    'HDFCBANK':    'Banking',
    'HDFCLIFE':    'Financial Services',
    'HDFC':        'Financial Services',
    'HEROMOTOCO':  'Automobile',
    'HINDALCO':    'Metals',
    'HINDUNILVR':  'Consumer Goods',
    'ICICIBANK':   'Banking',
    'ITC':         'Consumer Goods',
    'IOC':         'Energy',
    'INDUSINDBK':  'Banking',
    'INFY':        'Information Technology',
    'JSWSTEEL':    'Metals',
    'KOTAKBANK':   'Banking',
    'LT':          'Infrastructure',
    'MM':          'Automobile',
    'MARUTI':      'Automobile',
    'NTPC':        'Energy',
    'NESTLEIND':   'Consumer Goods',
    'ONGC':        'Energy',
    'POWERGRID':   'Energy',
    'RELIANCE':    'Energy',
    'SBILIFE':     'Financial Services',
    'SBIN':        'Banking',
    'SUNPHARMA':   'Pharmaceuticals',
    'TCS':         'Information Technology',
    'TATACONSUM':  'Consumer Goods',
    'TATAMOTORS':  'Automobile',
    'TATASTEEL':   'Metals',
    'TECHM':       'Information Technology',
    'TITAN':       'Consumer Goods',
    'ULTRACEMCO':  'Cement & Construction',
    'UPL':         'Agriculture',
    'WIPRO':       'Information Technology',
    'SHREECEM':    'Cement & Construction',
}

DISPLAY_NAMES = {
    'ADANIPORTS': 'Adani Ports',
    'ASIANPAINT': 'Asian Paints',
    'AXISBANK':   'Axis Bank',
    'BAJAJ-AUTO': 'Bajaj Auto',
    'BAJFINANCE': 'Bajaj Finance',
    'BAJAJFINSV': 'Bajaj Finserv',
    'BPCL':       'BPCL',
    'BHARTIARTL': 'Bharti Airtel',
    'BRITANNIA':  'Britannia',
    'CIPLA':      'Cipla',
    'COALINDIA':  'Coal India',
    'DIVISLAB':   'Divi\'s Labs',
    'DRREDDY':    'Dr. Reddy\'s',
    'EICHERMOT':  'Eicher Motors',
    'GRASIM':     'Grasim',
    'HCLTECH':    'HCL Tech',
    'HDFCBANK':   'HDFC Bank',
    'HDFCLIFE':   'HDFC Life',
    'HDFC':       'HDFC Ltd',
    'HEROMOTOCO': 'Hero MotoCorp',
    'HINDALCO':   'Hindalco',
    'HINDUNILVR': 'HUL',
    'ICICIBANK':  'ICICI Bank',
    'ITC':        'ITC',
    'IOC':        'Indian Oil',
    'INDUSINDBK': 'IndusInd Bank',
    'INFY':       'Infosys',
    'JSWSTEEL':   'JSW Steel',
    'KOTAKBANK':  'Kotak Bank',
    'LT':         'L&T',
    'MM':         'M&M',
    'MARUTI':     'Maruti Suzuki',
    'NTPC':       'NTPC',
    'NESTLEIND':  'Nestle India',
    'ONGC':       'ONGC',
    'POWERGRID':  'Power Grid',
    'RELIANCE':   'Reliance',
    'SBILIFE':    'SBI Life',
    'SBIN':       'State Bank of India',
    'SUNPHARMA':  'Sun Pharma',
    'TCS':        'TCS',
    'TATACONSUM': 'Tata Consumer',
    'TATAMOTORS': 'Tata Motors',
    'TATASTEEL':  'Tata Steel',
    'TECHM':      'Tech Mahindra',
    'TITAN':      'Titan',
    'ULTRACEMCO': 'UltraTech Cement',
    'UPL':        'UPL',
    'WIPRO':      'Wipro',
    'SHREECEM':   'Shree Cement',
}


def get_available_symbols():
    """Return list of symbols with available data files."""
    symbols = []
    for f in sorted(os.listdir(STOCKS_DIR)):
        if f.endswith('.csv'):
            symbols.append(f.replace('.csv', ''))
    return symbols


def load_stock(symbol: str, start_date: str = '2010-01-01', end_date: str = '2021-04-30') -> pd.DataFrame:
    """Load and clean a single stock's historical data."""
    path = os.path.join(STOCKS_DIR, f'{symbol}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data file for {symbol}")

    df = pd.read_csv(path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(' ', '_')
        if cl in ('close', 'close_price'):
            col_map[c] = 'Close'
        elif cl in ('open', 'open_price'):
            col_map[c] = 'Open'
        elif cl in ('high', 'high_price'):
            col_map[c] = 'High'
        elif cl in ('low', 'low_price'):
            col_map[c] = 'Low'
        elif cl in ('volume', 'traded_volume', 'total_traded_quantity'):
            col_map[c] = 'Volume'
        elif cl in ('vwap',):
            col_map[c] = 'VWAP'
    df = df.rename(columns=col_map)

    # Ensure required columns
    required = ['Date', 'Open', 'High', 'Low', 'Close']
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Column {r} not found in {symbol}")

    if 'Volume' not in df.columns:
        df['Volume'] = np.nan

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume'] +
            (['VWAP'] if 'VWAP' in df.columns else [])]

    # Filter date range
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df = df.dropna(subset=['Close', 'Open', 'High', 'Low'])
    df = df.reset_index(drop=True)

    df['Symbol']  = symbol
    df['Sector']  = SECTOR_MAP.get(symbol, 'Unknown')
    df['Name']    = DISPLAY_NAMES.get(symbol, symbol)

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators to a stock dataframe."""
    df = df.copy()
    close = df['Close']
    high  = df['High']
    low   = df['Low']
    vol   = df['Volume']

    # ── Returns ──────────────────────────────────────────────────────
    df['Daily_Return']  = close.pct_change()
    df['Log_Return']    = np.log(close / close.shift(1))

    # ── Moving Averages ──────────────────────────────────────────────
    for w in [5, 10, 20, 50, 100, 200]:
        df[f'MA_{w}']  = close.rolling(w).mean()
        df[f'EMA_{w}'] = close.ewm(span=w, adjust=False).mean()

    # ── Bollinger Bands (20-day, 2σ) ─────────────────────────────────
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std()
    df['BB_Upper']  = bb_mid + 2 * bb_std
    df['BB_Lower']  = bb_mid - 2 * bb_std
    df['BB_Mid']    = bb_mid
    df['BB_Width']  = (df['BB_Upper'] - df['BB_Lower']) / bb_mid
    df['BB_Pct']    = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)

    # ── RSI (14-day) ─────────────────────────────────────────────────
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist']   = df['MACD'] - df['MACD_Signal']

    # ── ATR (14-day Average True Range) ──────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(14).mean()

    # ── Stochastic Oscillator (%K, %D) ───────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ── OBV (On-Balance Volume) ───────────────────────────────────────
    if vol.notna().sum() > 10:
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        df['OBV'] = obv

    # ── Price-based features ─────────────────────────────────────────
    df['Price_Range']      = high - low
    df['Price_Range_Pct']  = df['Price_Range'] / close
    df['Gap_Up']           = (df['Open'] - close.shift(1)) / close.shift(1)

    # ── Volatility ───────────────────────────────────────────────────
    df['Volatility_20']  = df['Log_Return'].rolling(20).std() * np.sqrt(252)
    df['Volatility_60']  = df['Log_Return'].rolling(60).std() * np.sqrt(252)

    # ── Momentum ─────────────────────────────────────────────────────
    for lag in [5, 10, 20, 60]:
        df[f'Momentum_{lag}'] = close / close.shift(lag) - 1

    # ── Golden / Death Cross signals ─────────────────────────────────
    df['Golden_Cross'] = ((df['MA_50'] > df['MA_200']) &
                          (df['MA_50'].shift(1) <= df['MA_200'].shift(1))).astype(int)
    df['Death_Cross']  = ((df['MA_50'] < df['MA_200']) &
                          (df['MA_50'].shift(1) >= df['MA_200'].shift(1))).astype(int)

    # ── Target variable (next-day direction) ─────────────────────────
    df['Target_Direction'] = (close.shift(-1) > close).astype(int)
    df['Target_Return_5d'] = close.shift(-5) / close - 1

    return df


def load_all_stocks(symbols=None, start_date='2010-01-01', end_date='2021-04-30'):
    """Load all NIFTY-50 stocks and add technical indicators."""
    if symbols is None:
        symbols = get_available_symbols()
    all_dfs = []
    for sym in symbols:
        try:
            df = load_stock(sym, start_date, end_date)
            df = add_technical_indicators(df)
            all_dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not load {sym}: {e}")
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def build_returns_matrix(symbols=None, start_date='2010-01-01', end_date='2021-04-30'):
    """Build a wide returns matrix (dates × symbols)."""
    if symbols is None:
        symbols = get_available_symbols()
    frames = {}
    for sym in symbols:
        try:
            df = load_stock(sym, start_date, end_date)[['Date', 'Close']]
            df = df.drop_duplicates(subset=['Date'])
            df = df.set_index('Date')['Close']
            frames[sym] = df
        except Exception:
            pass
    prices = pd.DataFrame(frames).sort_index()
    prices = prices[~prices.index.duplicated(keep='first')]
    prices = prices.dropna(how='all')
    returns = prices.pct_change().dropna(how='all')
    return prices, returns


if __name__ == '__main__':
    syms = get_available_symbols()
    print(f"Available symbols ({len(syms)}): {syms[:5]} ...")
    df = load_stock('RELIANCE')
    df = add_technical_indicators(df)
    print(df[['Date', 'Close', 'RSI_14', 'MACD', 'BB_Upper', 'BB_Lower']].tail())
    print("Data loader OK")
