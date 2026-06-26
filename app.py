"""AI Stock Volatility Forecaster & Buy/Hold/Sell Recommender - Streamlit app.

Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub, then https://share.streamlit.io -> New app.

Research question: can AI (XGBoost / LSTM) using market data + news sentiment predict
forward realized volatility better than traditional models (GARCH, EWMA, naive)?
"""
import numpy as np
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="AI Stock Volatility", page_icon="📈", layout="wide")

st.title("📈 AI Stock Volatility Forecaster")
st.caption("Predicts forward realized volatility with XGBoost / LSTM, benchmarks them "
           "against GARCH · EWMA · naive, and turns the forecast into a BUY / HOLD / SELL "
           "call. Educational research tool — **not financial advice**.")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    n_tickers = st.slider("Number of stocks", 10, 250, 40, step=10,
                          help="More stocks = slower. Free hosting handles ~40 well.")
    years = st.slider("Years of daily history", 2, 8, 5)
    use_sentiment = st.checkbox("Use news sentiment (VADER)", value=True)
    use_lstm = st.checkbox("Also train LSTM (slow / heavy)", value=False,
                           help="TensorFlow is memory-heavy on free hosting. "
                                "XGBoost runs by default; enable this for the full AI vs "
                                "traditional comparison.")
    run = st.button("🚀 Run analysis", type="primary")
    st.markdown("---")
    st.caption("First run downloads data and trains models — give it a few minutes.")


# ---------------------------------------------------------------------------
# Cached pipeline (re-runs only when the settings above change)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def run_pipeline(n_tickers, years, use_sentiment, use_lstm):
    log = {}
    tickers = core.get_universe(n_tickers)
    prices = core.download_prices(tickers, years)
    feat = core.build_features(prices)
    feat = core.attach_sentiment(feat, use_sentiment)

    data, train, test, cutoff = core.split_by_date(feat)
    test = core.add_baselines(test)

    model, scaler, test, importances = core.train_xgb(train, test)

    # benchmark sample (so GARCH stays tractable) + GARCH fits
    sample = train["Ticker"].value_counts().head(min(40, n_tickers)).index.tolist()
    garch = core.garch_forecasts(train, sample)
    bench = test[test["Ticker"].isin(sample)].copy()
    bench["pred_garch"] = bench["Ticker"].map(garch)

    if use_lstm:
        lstm_train = train["Ticker"].value_counts().head(min(150, n_tickers)).index.tolist()
        _, lstm_series, _ = core.train_lstm(train, test, scaler, lstm_train, sample)
        test["pred_lstm"] = lstm_series.reindex(test.index)
        bench = bench.merge(test[["Date", "Ticker", "pred_lstm"]], on=["Date", "Ticker"], how="left")

    bench = bench.merge(test[["Date", "Ticker", "pred_xgb"]], on=["Date", "Ticker"], how="left")
    results = core.compare_models(bench)
    recs = core.recommend_table(data, model, scaler)

    log["n_stocks"] = data["Ticker"].nunique()
    log["cutoff"] = pd.Timestamp(cutoff).date()
    return dict(data=data, recs=recs, results=results, importances=importances, log=log)


if "bundle" not in st.session_state:
    st.session_state.bundle = None

if run:
    try:
        with st.spinner("Downloading data, training models, scoring stocks..."):
            st.session_state.bundle = run_pipeline(n_tickers, years, use_sentiment, use_lstm)
        st.success("Done!")
    except Exception as e:
        st.error("Something went wrong: %s" % e)
        st.stop()

bundle = st.session_state.bundle
if bundle is None:
    st.info("👈 Pick your settings and click **Run analysis** to begin.")
    st.stop()

recs = bundle["recs"]
results = bundle["results"]

# ---------------------------------------------------------------------------
# Top-line metrics
# ---------------------------------------------------------------------------
counts = recs["action"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks analyzed", bundle["log"]["n_stocks"])
c2.metric("BUY", int(counts.get("BUY", 0)))
c3.metric("HOLD", int(counts.get("HOLD", 0)))
c4.metric("SELL", int(counts.get("SELL", 0)))

tab1, tab2, tab3, tab4 = st.tabs(
    ["📋 Recommendations", "🔎 Single stock", "🏆 AI vs Traditional", "ℹ️ About"])

# --- Tab 1: recommendations table ------------------------------------------
with tab1:
    st.subheader("Buy / Hold / Sell across all stocks")
    pick = st.multiselect("Filter by action", ["BUY", "HOLD", "SELL"],
                          default=["BUY", "HOLD", "SELL"])
    view = recs[recs["action"].isin(pick)].copy()
    view["pred_volatility"] = (view["pred_volatility"] * 100).round(1).astype(str) + "%"
    view["momentum_21d"] = (view["momentum_21d"] * 100).round(1).astype(str) + "%"
    view["sentiment"] = view["sentiment"].round(3)
    st.dataframe(view, use_container_width=True, height=520)
    st.download_button("⬇️ Download recommendations (CSV)",
                       recs.to_csv(index=False).encode(),
                       file_name="stock_recommendations.csv", mime="text/csv")

# --- Tab 2: single stock ----------------------------------------------------
with tab2:
    st.subheader("Drill into one stock")
    sym = st.selectbox("Ticker", sorted(recs["Ticker"].unique()))
    row = recs[recs["Ticker"] == sym].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Recommendation", row["action"])
    m2.metric("Predicted volatility (annualized)", "%.1f%%" % (row["pred_volatility"] * 100))
    m3.metric("Risk", row["risk"])
    hist = bundle["data"]
    g = hist[hist["Ticker"] == sym].sort_values("Date").set_index("Date")
    st.line_chart(g[["Close"]], height=240)
    st.caption("Trailing 21-day volatility (annualized)")
    st.line_chart(g[["vol_21"]], height=200)

# --- Tab 3: model comparison ------------------------------------------------
with tab3:
    st.subheader("Does AI beat traditional finance models?")
    st.caption("Forecast error on the held-out test set (lower RMSE = better). "
               "All models scored on the same benchmark stocks.")
    show = results.copy()
    show["RMSE"] = show["RMSE"].round(5)
    show["MAE"] = show["MAE"].round(5)
    st.dataframe(show, use_container_width=True)
    st.bar_chart(results.set_index("Model")["RMSE"], height=300)

    ai = results[results["Model"].str.contains("AI")]["RMSE"].min()
    trad = results[results["Model"].isin(
        ["Naive (persistence)", "EWMA", "GARCH(1,1)"])]["RMSE"].min()
    if pd.notna(ai) and pd.notna(trad):
        verdict = "**beats**" if ai < trad else "**does NOT beat**"
        st.markdown("Best AI RMSE `%.5f` vs best traditional RMSE `%.5f` → AI %s the best "
                    "traditional model on this test set." % (ai, trad, verdict))
        st.caption("Re-run across several split dates before drawing firm conclusions.")
    st.markdown("##### XGBoost feature importance")
    st.bar_chart(bundle["importances"], height=300)

# --- Tab 4: about -----------------------------------------------------------
with tab4:
    st.markdown(
        "- **Target:** annualized realized volatility over the next 5 trading days.\n"
        "- **AI models:** XGBoost (always) and an optional LSTM neural network.\n"
        "- **Baselines:** GARCH(1,1), EWMA (λ=0.94), naive persistence.\n"
        "- **Sentiment:** keyless VADER scoring of recent yfinance headlines (a current, "
        "cross-sectional signal).\n"
        "- **Recommendation rule:** blends predicted volatility (risk), 21-day momentum "
        "(trend) and sentiment into BUY / HOLD / SELL.\n\n"
        "**Not financial advice.** Educational research tool. Data via yfinance changes daily."
    )
