import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(page_title="Frontera Eficiente - Markowitz", layout="wide")
st.title("Optimización de Portafolio - Markowitz con Yahoo Finance")

# ==========================================
# INPUTS
# ==========================================
st.sidebar.header("Parámetros")

tickers_input = st.sidebar.text_input(
    "Tickers (separados por coma)",
    value="AMZN,MSFT,GOOGL,BTC-USD,GC=F"
)

start_date = st.sidebar.date_input("Fecha inicio", pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("Fecha fin", pd.to_datetime("today"))

rf = st.sidebar.number_input("Tasa libre de riesgo", 0.0, 0.2, 0.05)

tickers = [t.strip() for t in tickers_input.split(",")]

# ==========================================
# BOTÓN
# ==========================================
if st.sidebar.button("Cargar datos"):

    # ======================================
    # DESCARGA
    # ======================================
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

    if data.empty:
        st.error("No se pudieron descargar datos. Verifica los tickers.")
        st.stop()

    data = data.dropna()

    st.subheader("Precios")
    st.dataframe(data.tail())

    # ======================================
    # RETORNOS (ROBUSTO)
    # ======================================
    returns = np.log(data / data.shift(1))
    returns = returns.dropna()

    # Eliminar columnas sin variación
    returns = returns.loc[:, returns.std() > 0]

    if returns.shape[1] < 2:
        st.error("No hay suficientes activos con datos válidos.")
        st.stop()

    # ======================================
    # MATRICES
    # ======================================
    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    tickers = returns.columns.tolist()

    # Validación crítica
    if len(mu) != len(tickers):
        st.error("Error de dimensiones en los datos.")
        st.stop()

    # ======================================
    # FUNCIONES
    # ======================================
    def portfolio_performance(w):
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
        return ret, vol

    def neg_sharpe(w):
        ret, vol = portfolio_performance(w)
        if vol == 0:
            return 999
        return -(ret - rf) / vol

    def volatility(w):
        return portfolio_performance(w)[1]

    # ======================================
    # OPTIMIZACIÓN
    # ======================================
    n = len(tickers)
    w0 = np.ones(n) / n

    bounds = tuple((0, 1) for _ in range(n))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    try:
        opt_sharpe = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        opt_min = minimize(
            volatility,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

    except Exception as e:
        st.error("Error en optimización")
        st.write(e)
        st.stop()

    w_sharpe = opt_sharpe.x
    w_min = opt_min.x

    # ======================================
    # FRONTERA EFICIENTE
    # ======================================
    target_returns = np.linspace(mu.min(), mu.max(), 50)
    frontier_vol = []

    for r in target_returns:
        constraints_eff = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mu) - r}
        )

        try:
            res = minimize(
                volatility,
                w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_eff
            )
            frontier_vol.append(res.fun)
        except:
            frontier_vol.append(np.nan)

    # ======================================
    # GRÁFICA
    # ======================================
    fig, ax = plt.subplots()

    ax.plot(frontier_vol, target_returns, 'g-', label="Frontera eficiente")

    ret_s, vol_s = portfolio_performance(w_sharpe)
    ret_m, vol_m = portfolio_performance(w_min)

    ax.scatter(vol_s, ret_s, c='red', s=200, label="Máx Sharpe")
    ax.scatter(vol_m, ret_m, c='blue', s=200, label="Mín Var")

    ax.set_xlabel("Volatilidad")
    ax.set_ylabel("Rendimiento")
    ax.set_title("Frontera Eficiente")
    ax.legend()

    st.pyplot(fig)

    # ======================================
    # RESULTADOS
    # ======================================
    st.subheader("Pesos óptimos")

    df_weights = pd.DataFrame({
        "Activo": tickers,
        "Max Sharpe": w_sharpe,
        "Min Var": w_min
    }).set_index("Activo")

    st.dataframe(df_weights.style.format("{:.2%}"))

    st.subheader("Métricas")

    st.write(f"Máx Sharpe → Retorno: {ret_s:.2%} | Volatilidad: {vol_s:.2%}")
    st.write(f"Mín Varianza → Retorno: {ret_m:.2%} | Volatilidad: {vol_m:.2%}")

    st.subheader("Correlaciones")
    st.dataframe(returns.corr())

else:
    st.info("Ingrese activos y presione 'Cargar datos'")
