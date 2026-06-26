"""Core logic for the AI Stock Volatility app.

Heavy ML libraries (xgboost, scikit-learn, arch, tensorflow) are imported *lazily*
inside the functions that need them, so this module can be imported and unit-tested
with only numpy + pandas available.

Research question: can AI (XGBoost / LSTM) using market data + news sentiment predict
forward realized volatility better than traditional models (GARCH, EWMA, naive)?
"""
import warnings, datetime as dt, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

# Annualization + feature definitions shared across the app -------------------
FEATURES = ["vol_5", "vol_10", "vol_21", "mom_5", "mom_10", "mom_21",
            "ma_ratio", "vol_chg", "rsi", "logret", "sentiment", "news_count"]

FALLBACK_TICKERS = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","BRK-B","JPM",
    "V","UNH","XOM","JNJ","WMT","MA","PG","HD","COST","ABBV","MRK","CVX","KO","PEP",
    "ADBE","AVGO","CRM","NFLX","AMD","INTC","CSCO","ACN","MCD","TMO","ABT","DHR","LIN",
    "NKE","TXN","PM","WFC","BAC","DIS","VZ","CMCSA","QCOM","HON","UNP","IBM","CAT","GE"]


# --- Universe ---------------------------------------------------------------
def get_universe(n=40):
    """Return up to n S&P 500 tickers (Wikipedia, with a hard-coded fallback)."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        syms = pd.read_html(url)[0]["Symbol"].astype(str).str.replace(".", "-", regex=False)
        tickers = [s.strip() for s in syms if s.strip()]
        if len(tickers) > 50:
            return tickers[:n]
    except Exception:
        pass
    return FALLBACK_TICKERS[:n]


# --- Price download ---------------------------------------------------------
def download_prices(tickers, years=5):
    import yfinance as yf
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years)
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    frames = []
    for t in tickers:
        try:
            sub = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except Exception:
            continue
        if sub is None or sub.empty or not {"Close", "Volume"}.issubset(sub.columns):
            continue
        sub = sub.dropna(how="all").reset_index()
        sub["Ticker"] = t
        frames.append(sub[["Date", "Ticker", "Close", "Volume"]])
    if not frames:
        raise RuntimeError("No price data downloaded - check connection / tickers.")
    out = pd.concat(frames, ignore_index=True).dropna(subset=["Close"])
    return out.sort_values(["Ticker", "Date"]).reset_index(drop=True)


# --- Features + target ------------------------------------------------------
def add_features(df, fwd_window=5):
    df = df.sort_values("Date").copy()
    r = df["Close"].pct_change()
    df["ret"] = r
    df["logret"] = np.log(df["Close"] / df["Close"].shift(1))
    for w in [5, 10, 21]:
        df["vol_%d" % w] = r.rolling(w).std() * np.sqrt(252)
        df["mom_%d" % w] = df["Close"].pct_change(w)
    df["ma_ratio"] = df["Close"] / df["Close"].rolling(20).mean()
    df["vol_chg"] = df["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))
    df["target_vol"] = r.shift(-1).rolling(fwd_window).std() * np.sqrt(252)
    return df


def build_features(prices, fwd_window=5):
    return pd.concat([add_features(g, fwd_window) for _, g in prices.groupby("Ticker")],
                     ignore_index=True)


# --- Sentiment (keyless VADER over yfinance headlines) ----------------------
def ticker_sentiment(ticker, analyzer):
    import yfinance as yf
    try:
        news = yf.Ticker(ticker).news or []
        scores = []
        for n in news:
            title = n.get("title")
            if not title and isinstance(n.get("content"), dict):
                title = n["content"].get("title")
            if title:
                scores.append(analyzer.polarity_scores(title)["compound"])
        if scores:
            return float(np.mean(scores)), len(scores)
    except Exception:
        pass
    return 0.0, 0


def attach_sentiment(feat, use_sentiment=True):
    if not use_sentiment:
        feat["sentiment"] = 0.0
        feat["news_count"] = 0
        return feat
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    smap, cmap = {}, {}
    for t in feat["Ticker"].unique():
        s, c = ticker_sentiment(t, analyzer)
        smap[t], cmap[t] = s, c
        time.sleep(0.02)
    feat["sentiment"] = feat["Ticker"].map(smap).fillna(0.0)
    feat["news_count"] = feat["Ticker"].map(cmap).fillna(0)
    return feat


# --- Train / test split -----------------------------------------------------
def split_by_date(feat, test_fraction=0.2):
    data = feat.dropna(subset=FEATURES + ["target_vol"]).copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["target_vol"])
    cutoff = data["Date"].quantile(1 - test_fraction)
    return data, data[data["Date"] < cutoff].copy(), data[data["Date"] >= cutoff].copy(), cutoff


# --- Traditional baselines --------------------------------------------------
def ewma_vol(returns, lam=0.94, ann=None):
    ann = ann if ann is not None else np.sqrt(252)
    out = np.zeros(len(returns))
    var = np.nanvar(returns[:20]) if len(returns) > 20 else np.nanvar(returns)
    for i, x in enumerate(returns):
        var = lam * var + (1 - lam) * (0 if np.isnan(x) else x) ** 2
        out[i] = np.sqrt(var) * ann
    return out


def add_baselines(test):
    test = test.copy()
    test["pred_naive"] = test["vol_21"]
    parts = []
    for _, g in test.groupby("Ticker"):
        g = g.sort_values("Date")
        parts.append(pd.Series(ewma_vol(g["ret"].values), index=g.index))
    test["pred_ewma"] = pd.concat(parts) if parts else np.nan
    return test


def garch_forecasts(train, sample_tickers, fwd_window=5):
    from arch import arch_model
    preds = {}
    for t in sample_tickers:
        try:
            tr = train[train["Ticker"] == t].sort_values("Date")["ret"].dropna() * 100
            if len(tr) < 100:
                continue
            res = arch_model(tr, p=1, q=1, mean="constant", vol="GARCH").fit(disp="off")
            fc = res.forecast(horizon=fwd_window, reindex=False)
            preds[t] = np.sqrt(fc.variance.values[-1].mean()) / 100 * np.sqrt(252)
        except Exception:
            continue
    return preds


# --- XGBoost ----------------------------------------------------------------
def train_xgb(train, test):
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    scaler = StandardScaler().fit(train[FEATURES])
    model = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42)
    model.fit(scaler.transform(train[FEATURES]), train["target_vol"].values)
    test = test.copy()
    test["pred_xgb"] = model.predict(scaler.transform(test[FEATURES]))
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    return model, scaler, test, importances


# --- LSTM (optional, heavy) -------------------------------------------------
def make_sequences(df, tickers, scaler, seq_len=20):
    Xs, ys, idx = [], [], []
    for t in tickers:
        g = df[df["Ticker"] == t].sort_values("Date")
        if len(g) <= seq_len:
            continue
        fv = scaler.transform(g[FEATURES]); tv = g["target_vol"].values; gi = g.index.values
        for i in range(seq_len, len(g)):
            Xs.append(fv[i - seq_len:i]); ys.append(tv[i]); idx.append(gi[i])
    if not Xs:
        return (np.empty((0, seq_len, len(FEATURES))), np.empty((0,)), np.array([]))
    return np.array(Xs), np.array(ys), np.array(idx)


def train_lstm(train, test, scaler, train_tickers, test_tickers, seq_len=20, epochs=10):
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    tf.random.set_seed(42)
    Xtr, ytr, _ = make_sequences(train, train_tickers, scaler, seq_len)
    Xte, yte, idx_te = make_sequences(test, test_tickers, scaler, seq_len)
    model = Sequential([Input(shape=(seq_len, len(FEATURES))),
                        LSTM(64, return_sequences=True), Dropout(0.2),
                        LSTM(32), Dropout(0.2),
                        Dense(16, activation="relu"), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    if len(Xtr):
        model.fit(Xtr, ytr, validation_split=0.1, epochs=epochs, batch_size=256, verbose=0)
    preds = model.predict(Xte, verbose=0).ravel() if len(Xte) else np.array([])
    series = pd.Series(preds, index=idx_te) if len(preds) else pd.Series(dtype=float)
    return model, series, (yte, preds)


# --- Model comparison (answers the research question) -----------------------
def compare_models(bench):
    cols = {"Naive (persistence)": "pred_naive", "EWMA": "pred_ewma",
            "GARCH(1,1)": "pred_garch", "XGBoost (AI)": "pred_xgb", "LSTM (AI)": "pred_lstm"}
    rows = []
    for name, col in cols.items():
        if col not in bench.columns:
            continue
        d = bench.dropna(subset=[col, "target_vol"])
        if len(d) == 0:
            continue
        err = d[col].values - d["target_vol"].values
        rows.append({"Model": name, "N": len(d),
                     "RMSE": float(np.sqrt(np.mean(err ** 2))),
                     "MAE": float(np.mean(np.abs(err)))})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


# --- Recommendation engine --------------------------------------------------
def recommend_table(data, model, scaler):
    latest = data.sort_values("Date").groupby("Ticker").tail(1).copy()
    latest["pred_vol"] = model.predict(scaler.transform(latest[FEATURES]))
    hi = latest["pred_vol"].quantile(0.66)
    lo = latest["pred_vol"].quantile(0.33)

    def _row(row):
        trend = np.sign(row["mom_21"])
        score = 0.6 * trend + 0.4 * np.sign(row["sentiment"])
        high_risk = row["pred_vol"] >= hi
        if score > 0.3 and not high_risk:
            action = "BUY"
        elif score < -0.1 or (high_risk and trend < 0):
            action = "SELL"
        else:
            action = "HOLD"
        risk = "HIGH" if high_risk else ("LOW" if row["pred_vol"] <= lo else "MED")
        return pd.Series({"signal_score": round(float(score), 3), "risk": risk, "action": action})

    recs = latest.join(latest.apply(_row, axis=1))
    out = recs[["Ticker", "Close", "mom_21", "sentiment", "pred_vol", "risk", "signal_score", "action"]]
    out = out.rename(columns={"mom_21": "momentum_21d", "pred_vol": "pred_volatility"})
    return out.sort_values(["action", "signal_score"], ascending=[True, False]).reset_index(drop=True)
