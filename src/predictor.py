"""
Stock Predictor Engine
NIFTY-50 Investment Intelligence Platform
Models: Random Forest, Gradient Boosting, Linear Regression, LSTM-like MLP
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, mean_absolute_error,
                              mean_squared_error, r2_score)
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings, os, pickle
warnings.filterwarnings('ignore')

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    'MA_5', 'MA_10', 'MA_20', 'MA_50',
    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
    'BB_Width', 'BB_Pct',
    'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'ATR_14',
    'Stoch_K', 'Stoch_D',
    'Volatility_20', 'Volatility_60',
    'Momentum_5', 'Momentum_10', 'Momentum_20',
    'Price_Range_Pct',
    'Daily_Return',
    'Log_Return',
    'Gap_Up',
]


# ─────────────────────────────────────────────────────────────────────────────
#  Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame, lag_days: int = 5):
    """
    Build the feature matrix X and targets y from a stock dataframe.
    Adds lagged versions of key features for temporal context.
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)

    avail_feats = [c for c in FEATURE_COLS if c in df.columns]

    # Add lag features
    lag_features = ['Daily_Return', 'RSI_14', 'MACD', 'Volatility_20', 'BB_Pct']
    lag_feats_avail = [c for c in lag_features if c in df.columns]
    for col in lag_feats_avail:
        for lag in range(1, lag_days + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
            avail_feats.append(f'{col}_lag{lag}')

    # Close ratio features
    df['Close_to_MA20']  = df['Close'] / df['MA_20']
    df['Close_to_MA50']  = df['Close'] / df['MA_50']
    df['MA20_to_MA50']   = df['MA_20'] / df['MA_50']
    for c in ['Close_to_MA20', 'Close_to_MA50', 'MA20_to_MA50']:
        if c not in avail_feats:
            avail_feats.append(c)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=avail_feats + ['Target_Direction', 'Target_Return_5d'])
    df = df.reset_index(drop=True)

    X = df[avail_feats].values
    y_class  = df['Target_Direction'].values.astype(int)
    y_return = df['Target_Return_5d'].values
    dates    = df['Date'].values

    return X, y_class, y_return, dates, avail_feats


# ─────────────────────────────────────────────────────────────────────────────
#  Model training
# ─────────────────────────────────────────────────────────────────────────────

def train_direction_model(X, y, model_type='rf'):
    """Train a classification model for next-day price direction."""
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
    elif model_type == 'gb':
        model = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
    elif model_type == 'mlp':
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            max_iter=500, random_state=42, early_stopping=True,
            learning_rate_init=0.001
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.fit(X, y)
    return model


def train_return_model(X, y, model_type='rf'):
    """Train a regression model for 5-day forward return."""
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
    elif model_type == 'gb':
        model = GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        )
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'mlp':
        model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), activation='relu',
            max_iter=500, random_state=42, early_stopping=True,
            learning_rate_init=0.001
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model.fit(X, y)
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Time-series cross-validation evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_tscv(X, y_class, y_return, n_splits=5):
    """
    Walk-forward time-series cross-validation.
    Returns averaged metrics across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    clf_metrics = []
    reg_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < 100 or len(test_idx) < 20:
            continue

        X_tr, X_te = X[train_idx], X[test_idx]
        yc_tr, yc_te = y_class[train_idx], y_class[test_idx]
        yr_tr, yr_te = y_return[train_idx], y_return[test_idx]

        # Scale
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        # Classification
        clf = train_direction_model(X_tr_s, yc_tr, 'rf')
        yc_pred = clf.predict(X_te_s)
        clf_metrics.append({
            'Fold': fold + 1,
            'Accuracy':  round(accuracy_score(yc_te, yc_pred) * 100, 2),
            'Precision': round(precision_score(yc_te, yc_pred, zero_division=0) * 100, 2),
            'Recall':    round(recall_score(yc_te, yc_pred, zero_division=0) * 100, 2),
            'F1':        round(f1_score(yc_te, yc_pred, zero_division=0) * 100, 2),
        })

        # Regression
        reg = train_return_model(X_tr_s, yr_tr, 'rf')
        yr_pred = reg.predict(X_te_s)
        clf_metrics[-1].update({
            'MAE':  round(mean_absolute_error(yr_te, yr_pred) * 100, 4),
            'RMSE': round(np.sqrt(mean_squared_error(yr_te, yr_pred)) * 100, 4),
            'R2':   round(r2_score(yr_te, yr_pred), 4),
        })

    df_metrics = pd.DataFrame(clf_metrics)
    return df_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  Full training pipeline for one stock
# ─────────────────────────────────────────────────────────────────────────────

def train_stock_model(df: pd.DataFrame, symbol: str, model_type: str = 'rf'):
    """
    Full pipeline: feature engineering → scaling → training → evaluation.
    Returns trained models, scaler, feature names, and metrics.
    """
    X, y_class, y_return, dates, feat_names = prepare_features(df)

    if len(X) < 200:
        raise ValueError(f"Insufficient data for {symbol}: {len(X)} rows")

    scaler = StandardScaler()

    # Train/test split — last 20% as test (time-based)
    split = int(len(X) * 0.80)
    X_tr, X_te   = X[:split], X[split:]
    yc_tr, yc_te = y_class[:split], y_class[split:]
    yr_tr, yr_te = y_return[:split], y_return[split:]

    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Direction model
    clf = train_direction_model(X_tr_s, yc_tr, model_type)
    yc_pred = clf.predict(X_te_s)

    # Return model
    reg = train_return_model(X_tr_s, yr_tr, model_type)
    yr_pred = reg.predict(X_te_s)

    # Metrics
    metrics = {
        'Symbol':    symbol,
        'Model':     model_type,
        'Train_Rows': split,
        'Test_Rows':  len(X_te),
        'Accuracy':   round(accuracy_score(yc_te, yc_pred) * 100, 2),
        'Precision':  round(precision_score(yc_te, yc_pred, zero_division=0) * 100, 2),
        'Recall':     round(recall_score(yc_te, yc_pred, zero_division=0) * 100, 2),
        'F1':         round(f1_score(yc_te, yc_pred, zero_division=0) * 100, 2),
        'MAE_5d':     round(mean_absolute_error(yr_te, yr_pred) * 100, 4),
        'RMSE_5d':    round(np.sqrt(mean_squared_error(yr_te, yr_pred)) * 100, 4),
        'R2_5d':      round(r2_score(yr_te, yr_pred), 4),
    }

    # Feature importances (for RF)
    importances = {}
    if hasattr(clf, 'feature_importances_'):
        importances = dict(zip(feat_names, clf.feature_importances_.tolist()))

    # Predictions on test set
    test_preds = pd.DataFrame({
        'Date':          dates[split:],
        'Actual_Dir':    yc_te,
        'Pred_Dir':      yc_pred,
        'Actual_Ret5d':  yr_te,
        'Pred_Ret5d':    yr_pred,
        'Dir_Prob_Up':   clf.predict_proba(X_te_s)[:, 1] if hasattr(clf, 'predict_proba') else yr_pred * 0,
    })

    return {
        'clf':          clf,
        'reg':          reg,
        'scaler':       scaler,
        'feat_names':   feat_names,
        'metrics':      metrics,
        'importances':  importances,
        'test_preds':   test_preds,
    }


def save_model(result: dict, symbol: str):
    path = os.path.join(MODELS_DIR, f'{symbol}_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(result, f)
    return path


def load_model(symbol: str) -> dict:
    path = os.path.join(MODELS_DIR, f'{symbol}_model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model for {symbol}")
    with open(path, 'rb') as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
#  Predict on latest data
# ─────────────────────────────────────────────────────────────────────────────

def predict_latest(df: pd.DataFrame, result: dict) -> dict:
    """
    Generate prediction for the most recent row of a stock dataframe.
    """
    X, _, _, dates, _ = prepare_features(df)
    feat_names = result['feat_names']
    # Re-build features aligned with training
    X_full, _, _, dates_full, all_feats = prepare_features(df)
    # Select only training features
    df2 = df.copy().sort_values('Date').reset_index(drop=True)

    # Re-prepare with same feature set
    avail = [c for c in feat_names if c in df2.columns or 'lag' in c]

    # Use last row
    X_latest = X_full[-1:, :]
    # Map to training feature indices
    if X_latest.shape[1] != len(feat_names):
        # Fallback: use available features only
        X_latest = X_full[-1:, :len(feat_names)]

    scaler = result['scaler']
    clf    = result['clf']
    reg    = result['reg']

    try:
        X_s = scaler.transform(X_latest)
        direction_prob = clf.predict_proba(X_s)[0][1] if hasattr(clf, 'predict_proba') else 0.5
        direction      = int(clf.predict(X_s)[0])
        ret_5d         = float(reg.predict(X_s)[0])
    except Exception as e:
        return {'error': str(e)}

    signal = 'STRONG BUY' if direction_prob > 0.65 else \
             'BUY'        if direction_prob > 0.55 else \
             'NEUTRAL'    if direction_prob > 0.45 else \
             'SELL'       if direction_prob > 0.35 else 'STRONG SELL'

    return {
        'Date':           str(dates_full[-1])[:10],
        'Direction':      'UP' if direction else 'DOWN',
        'Direction_Prob': round(direction_prob * 100, 1),
        'Pred_Return_5d': round(ret_5d * 100, 2),
        'Signal':         signal,
        'Confidence':     'High' if abs(direction_prob - 0.5) > 0.15 else 'Medium' if abs(direction_prob - 0.5) > 0.07 else 'Low',
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Batch training for all stocks
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(all_df: pd.DataFrame, model_type='rf', save=True):
    """Train models for every symbol and return summary metrics."""
    summary = []
    for symbol, gdf in all_df.groupby('Symbol'):
        try:
            result = train_stock_model(gdf.copy(), symbol, model_type)
            if save:
                save_model(result, symbol)
            summary.append(result['metrics'])
            print(f"  [{symbol}] Acc={result['metrics']['Accuracy']}%  R2={result['metrics']['R2_5d']}")
        except Exception as e:
            print(f"  Warning [{symbol}]: {e}")
    return pd.DataFrame(summary)


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_stock, add_technical_indicators
    df = load_stock('INFY')
    df = add_technical_indicators(df)
    result = train_stock_model(df, 'INFY')
    print("=== INFY Model Metrics ===")
    for k, v in result['metrics'].items():
        print(f"  {k}: {v}")
    print("\n=== Top 10 Features ===")
    top = sorted(result['importances'].items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in top:
        print(f"  {feat}: {imp:.4f}")
    pred = predict_latest(df, result)
    print("\n=== Latest Prediction ===")
    for k, v in pred.items():
        print(f"  {k}: {v}")
