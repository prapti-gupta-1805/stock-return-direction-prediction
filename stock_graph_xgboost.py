import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. SETUP
stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ITC.NS", "LT.NS", "SBIN.NS", "AXISBANK.NS", "HINDUNILVR.NS"]

print("Downloading Data...")
data_raw = yf.download(stocks + ["^NSEI", "^VIX"], start="2017-01-01", end="2024-12-31", progress=False)
closes = data_raw['Close'].ffill()
highs = data_raw['High'].ffill()
lows = data_raw['Low'].ffill()
nifty = closes['^NSEI']

def get_refined_features(ticker):
    p = closes[ticker].dropna()
    h = highs[ticker].dropna()
    l = lows[ticker].dropna()
    
    # Target: 5-Day Forward Return vs Market
    s_ret = np.log(p / p.shift(5)).shift(-5)
    m_ret = np.log(nifty / nifty.shift(5)).shift(-5)
    s_ret, m_ret = s_ret.align(m_ret, join='inner')
    
    df = pd.DataFrame(index=s_ret.index)
    df['Target'] = (s_ret > m_ret).astype(int)
    
    # Feature 1: ATR-Normalized Volatility (Price Range relative to Vol)
    atr = ta.volatility.average_true_range(h, l, p, window=14)
    df['ATR_Norm'] = (p - p.rolling(14).mean()) / atr
    
    # Feature 2: Distance from Bollinger Bands
    indicator_bb = ta.volatility.BollingerBands(close=p, window=20, window_dev=2)
    df['BB_High_Dist'] = (p - indicator_bb.bollinger_hband()) / p
    df['BB_Low_Dist'] = (p - indicator_bb.bollinger_lband()) / p
    
    # Feature 3: Money Flow Index (Pressure)
    # Note: Requires volume, but using RSI as proxy for speed here
    df['RSI'] = ta.momentum.rsi(p, window=14)
    
    # Feature 4: Market Regime (Is Nifty above 200MA?)
    df['Market_Regime'] = (nifty > nifty.rolling(200).mean()).astype(int)

    return df.dropna()

# 2. HYBRID MODEL TRAINING
print("Training Hybrid Ensemble per stock...")
all_preds = []
all_actuals = []

for s in stocks:
    df_s = get_refined_features(s)
    
    split = int(len(df_s) * 0.85)
    train, test = df_s.iloc[:split], df_s.iloc[split:]
    
    X_train, y_train = train.drop('Target', axis=1), train['Target']
    X_test, y_test = test.drop('Target', axis=1), test['Target']
    
    # Model A: XGBoost (Gradient Boosting)
    xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42, eval_metric='logloss')
    
    # Model B: Random Forest (Bagging - better at handling noise)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
    
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    
    # Soft Voting: Average the probabilities
    prob_avg = (xgb.predict_proba(X_test)[:, 1] + rf.predict_proba(X_test)[:, 1]) / 2
    
    # Use a dynamic threshold based on training mean
    threshold = y_train.mean() 
    preds = (prob_avg > threshold).astype(int)
    
    all_preds.extend(preds)
    all_actuals.extend(y_test)
    print(f"Ticker: {s.ljust(12)} | Acc: {accuracy_score(y_test, preds):.4f}")

print("\n" + "="*40)
print(f"FINAL HYBRID ACCURACY: {accuracy_score(all_actuals, all_preds):.4f}")
print("="*40)
print(classification_report(all_actuals, all_preds))