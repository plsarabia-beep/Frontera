import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Frontera Eficiente - Markowitz", layout="wide")

st.title("Optimización de Portafolio con Yahoo Finance")

st.sidebar.header("Parámetros")

tickers_input = st.sidebar.text_input(
    "Tickers (separados por coma)",
    value="AMZN,MSFT,GOOGL,BTC-USD,GC=F"
)

start_date = st.sidebar.date_input("Fecha inicio", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("Fecha fin", value=pd.to_datetime("today"))

rf = st.sidebar.number_input("Tasa libre de riesgo", 0.0, 0.2, 0.05)

tickers = [t.strip() for t in tickers_input.split(",")]

if st.sidebar.button("Cargar datos"):

    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]
    data = data.dropna()

    st.subheader("Precios históricos")
    st.dataframe(data.tail())

    returns = np.log(data / data.shift(1)).dropna()

    mu = returns.mean() * 252
    cov = returns.cov() * 252

    def portfolio_performance(w):
        ret = np.dot(w, mu)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return ret, vol

    def neg_sharpe(w):
        ret, vol = portfolio_performance(w)
        return -(ret - rf) / vol

    def volatility(w):
        return portfolio_performance(w)[1]

    n = len(tickers)
    w0 = np.ones(n) / n

    bounds = tuple((0,1) for _ in range(n))
    constraints = ({'type':'eq','fun':lambda x: np.sum(x)-1})

    opt_sharpe = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_min = minimize(volatility, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    w_sharpe = opt_sharpe.x
    w_min = opt_min.x

    target_returns = np.linspace(mu.min(), mu.max(), 60)
    frontier_vol = []

    for r in target_returns:
        constraints_eff = (
            {'type':'eq','fun':lambda x: np.sum(x)-1},
            {'type':'eq','fun':lambda x: np.dot(x, mu) - r}
        )

        res = minimize(volatility, w0, method='SLSQP', bounds=bounds, constraints=constraints_eff)
        frontier_vol.append(res.fun)

    fig, ax = plt.subplots()
    ax.plot(frontier_vol, target_returns, 'g-', label="Frontera eficiente")

    ret_s, vol_s = portfolio_performance(w_sharpe)
    ret_m, vol_m = portfolio_performance(w_min)

    ax.scatter(vol_s, ret_s, c='red', s=200, label="Máx Sharpe")
    ax.scatter(vol_m, ret_m, c='blue', s=200, label="Mín Var")

    ax.set_xlabel("Volatilidad")
    ax.set_ylabel("Rendimiento")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Pesos óptimos")

    df_weights = pd.DataFrame({
        "Activo": tickers,
        "Max Sharpe": w_sharpe,
        "Min Var": w_min
    }).set_index("Activo")

    st.dataframe(df_weights.style.format("{:.2%}"))

else:
    st.info("Definir activos y cargar datos")
