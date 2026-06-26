# 📈 AI Stock Volatility Forecaster

A web app that predicts **stock volatility** with AI (XGBoost + optional LSTM), benchmarks
it against traditional finance models (**GARCH, EWMA, naive**), and gives a
**BUY / HOLD / SELL** call for hundreds of stocks using market data **and news sentiment**.

> **Research question:** Can AI predict stock-market variability better than traditional
> finance models by using market data and news sentiment?

> ⚠️ Educational research project — **not financial advice**.

---

## What it does

- Pulls daily prices for S&P 500 stocks (yfinance) + keyless VADER news sentiment.
- Forecasts the next 5 trading days' realized volatility.
- Compares **XGBoost / LSTM (AI)** vs **GARCH / EWMA / naive (traditional)** by RMSE/MAE.
- Ranks every stock BUY / HOLD / SELL from volatility + momentum + sentiment.
- Interactive: filter, drill into a single stock, download the results as CSV.

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501).

---

## Deploy as a public web app (free, GitHub → Streamlit Cloud)

1. **Create a GitHub repo** and upload these files (`app.py`, `core.py`,
   `requirements.txt`, `README.md`). Easiest way with no command line:
   - On github.com click **New repository** → name it e.g. `stock-volatility-ai` → **Create**.
   - Click **uploading an existing file**, drag in all the files, then **Commit changes**.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app**, pick your repo, set **Main file path** to `app.py`, click **Deploy**.
4. Wait a few minutes for the first build. You'll get a public link like
   `https://your-app.streamlit.app` you can share.

### Enabling the LSTM
The LSTM is off by default because TensorFlow is memory-heavy. To enable it:
1. Uncomment `tensorflow-cpu>=2.15` in `requirements.txt`.
2. Redeploy, then tick **"Also train LSTM"** in the app sidebar.
If the free tier runs out of memory, leave TensorFlow off and use XGBoost only (the app
still answers the research question against the traditional baselines).

---

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit user interface |
| `core.py` | Data, features, sentiment, models, baselines, recommender (testable) |
| `requirements.txt` | Dependencies |

## Performance tips
- Start with ~40 stocks; raise the slider once it's working.
- Results are cached — changing a setting re-runs; clicking around does not.
- yfinance data and live news change daily, so re-runs can differ.
