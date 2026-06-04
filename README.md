# S&P 500 Forecasting and Econometrics

Two studies on the S&P 500, sharing one ticker (`^GSPC`).

1. **`LLaMA_P1_P2_P3_forecasting.ipynb`** — can an 8B LLM forecast the next-day close better than a random walk, AR(1) and ARIMAX, and does a Google Trends attention signal help?
2. **`SP500_Financial_Econometrics_Analysis.ipynb`** — is the S&P 500 weak-form efficient? Stationarity, returns distribution, the random-walk hypothesis, and volatility clustering.

---

## Notebook 1 — S&P 500 financial econometrics

A self-contained study of the daily return series, in roughly the order you'd actually want to ask the questions.

It starts with stationarity, running ADF and KPSS. Then the returns distribution: histogram against a fitted normal, Q-Q plot, and three normality tests (Shapiro–Wilk, Anderson–Darling, Kolmogorov–Smirnov).

The weak-form efficiency section is the core. Ljung–Box on raw returns *and* on squared returns, ARCH-LM, Durbin–Watson on AR(1) residuals, and a Lo–MacKinlay variance ratio test with the heteroskedasticity-robust z-statistic.


Runs end to end on `yfinance` data with no GPU and no API keys.

```bash
pip install -r requirements.txt
jupyter lab
```

---

## Notebook 2 — LLaMA prompting vs. statistical baselines

Rolling-window, one-step-ahead forecasting of the next-day S&P 500 close. Three LLaMA-3.1-8B prompting strategies against three statistical baselines, with Diebold–Mariano tests to check whether any gap is real or just sampling noise.

Retail attention: a z-scored Google Trends composite from four search topics (`S&P 500`, `Bursa de valori`, `Yahoo Finance`, `Recesiune`). The question isn't only "can the LLM forecast prices" but "does feeding it an attention signal change anything." The whole sample (Jan 2024 – May 2026) sits after LLaMA 3.1's pretraining cutoff, so the model can't have memorized these prices.

### Reproductibility — use `trends_weekly.csv`

**For full reproductibility of the results, run this notebook with the bundled `trends_weekly.csv`.** It's the cached weekly Google Trends series the entire experiment is built on.

The notebook fetches Trends through SerpAPI. Google Trends returns slightly different numbers on every query because it samples, so even the notebook averages five independent draws to smooth that out. Re-fetching live would give you a *different* attention series than the one behind the results.

How the cache is picked up: the setup cell looks for `trends_weekly.csv` in the working directory.

```python
WEEKLY_CACHE = Path("trends_weekly.csv")
if WEEKLY_CACHE.exists():
    # loads the cache, no SerpAPI call
elif SERPAPI_KEY:
    # re-fetches live (different numbers, needs a key)
```

So the rule is simple. On Colab, upload `trends_weekly.csv` to `/content` (the working directory) before running. Running locally, launch Jupyter from the repo root so the relative path resolves. If the cache is present, no SerpAPI key is needed and the attention input is identical to the one used here.

One honest caveat: the cache pins the *attention input* exactly, which is the part you control. The LLM still samples at generation time (`do_sample=True`, `temperature=0.2`, 20 draws per prediction, median taken), so LLM metrics won't be bit-identical across machines and GPUs. The statistical baselines are deterministic and reproduce exactly. The seeds (`numpy`, `torch`, `set_seed(42)`) reduce LLM variance but don't eliminate cross-hardware differences in sampling.

### What's compared

Six models, all producing a next-day price level so the metrics line up:

- `random_walk` — predict last close.
- `ar1` — AR(1) on log-returns.
- `arimax_attn` — `auto_arima` on log-returns with lag-1 attention as an exogenous regressor.
- `p1` — LLaMA, prices only, zero-shot.
- `p2` — LLaMA, prices and yesterday's z-scored attention, zero-shot.
- `p3` — LLaMA, few-shot: three in-context examples pairing a price window and its attention reading with the realized next-day price.

Rolling window of 15 trading days, step size 5, roughly 115 one-step-ahead predictions.

### How it's evaluated

RMSE in price space and R². Then Diebold–Mariano with the Harvey–Leybourne–Newbold small-sample correction for pairwise tests of equal predictive accuracy, plus a separate pass restricted to the Feb–Jun 2025 drawdown to see how the models behave under stress.

Before any of that, the notebook runs a sanity check. It correlates lag-1 attention against forward returns and fits full-sample AR(1) and ARIMAX models with heteroskedasticity-robust standard errors, with Ljung–Box, ARCH-LM, and Jarque–Bera diagnostics on the residuals.

### Setup

Designed for Google Colab on a T4 GPU. You need a Hugging Face token with access to the gated `meta-llama/Llama-3.1-8B-Instruct`:

```python
# Colab: Secrets panel
HF_TOKEN     # required — gated model access
SERPAPI_KEY  # optional — only if you re-fetch Trends instead of using the cache
```

Locally:

```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here   # SERPAPI_KEY not needed when the CSV is present
jupyter lab                       # from the repo root, so trends_weekly.csv resolves
```

A GPU is required. The notebook raises an error if CUDA isn't available.

---

## Files

- `SP500_Financial_Econometrics_Analysis.ipynb` — the econometrics study (Notebook 1).
- `LLaMA_P1_P2_P3_forecasting.ipynb` — the LLM forecasting pipeline (Notebook 2).
- `trends_weekly.csv` — cached weekly Google Trends composite. Required for Notebook 2's reproducibility.
- `requirements.txt`.
