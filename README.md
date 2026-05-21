# Bayesian Time-Varying Coefficient Model (BTVC / KTR)

Project for ME607 — Time Series Analysis at UNICAMP (2026).

The goal was to study and implement the BTVC model proposed by Ng, Wang & Dai (2021), which was originally developed at Uber to improve Marketing Mix Modeling. The core idea is that regression coefficients don't need to be fixed — they can evolve smoothly over time using kernel weighting.

I applied it to the `iclaims` dataset: 443 weeks of US unemployment insurance claims (2010–2018) with Google Trends regressors.

## What the code does

- Exploratory analysis of the series and regressors
- Manual ACF and PACF implementation (Durbin-Levinson recursion) to assess stationarity
- KTR model fitting via Orbit with Stochastic Variational Inference (Pyro)
- Signal decomposition: trend + seasonality + regression components
- Time-varying coefficient plots for each regressor
- Residual diagnostics
- Forecast evaluation: SMAPE ≈ 0.78%, MAE ≈ 0.096 on a 52-week holdout

## Why BTVC over ARIMA/SARIMA

Classical models assume fixed parameters. The `iclaims` series has a structural break in the `trend.unemploy` regressor around 2014–2015 (Google Trends reindexing), which would distort a fixed-coefficient model. The BTVC handles this naturally by adjusting β(t) before and after the break without affecting other components.

## Stack

Python, Orbit (orbit-ml), Pyro, Pandas, NumPy, Matplotlib, Statsmodels

```bash
pip install orbit-ml matplotlib pandas numpy scipy statsmodels
python ME607_codigo_236312_thales.py
```

## Reference

Ng, Wang & Dai (2021). *Bayesian Time Varying Coefficient Model with Applications to Marketing Mix Modeling.* [arXiv:2106.03322](https://arxiv.org/abs/2106.03322)
