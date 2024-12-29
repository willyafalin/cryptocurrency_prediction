# -------------------------------------------------
# IMPORTS
# -------------------------------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from binance.client import Client
from datetime import datetime, timedelta
import function2  # Custom functions for data handling and model predictions
import streamlit as st

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def fetch_data(crypto_pairs, interval, start_date, end_date):
    """Fetch historical data for multiple cryptocurrencies."""
    data_dict = {}
    for pair in crypto_pairs:
        data = function2.fetch_binance_data_full(pair, interval, start_date, end_date)
        data_dict[pair] = data
    return data_dict

def calculate_volatility(data_dict, interval):
    """Calculate GARCH-based volatility for each cryptocurrency."""
    volatilities = []
    for pair, data in data_dict.items():
        p, q, dist = (2, 2, 't') if interval == '1h' else (1, 1, 'normal')
        garch_result = function2.garch_multi_forecast(
            data=data, p=p, q=q, dist=dist, horizon=1, window_realized=10
        )
        col_name = 'predicted_volatility_original_1'
        vol = garch_result[col_name].dropna().iloc[-1] if col_name in garch_result.columns else 0.0
        volatilities.append(vol)
    return np.array(volatilities)

def covariance_matrix(vol_array):
    n = len(vol_array)
    corr_matrix = np.identity(n)  # Assuming perfect correlation for simplicity
    cov_matrix = np.outer(vol_array, vol_array) * corr_matrix
    return cov_matrix

def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.01):
    n_assets = len(expected_returns)
    init_weights = np.ones(n_assets) / n_assets

    def sharpe_ratio(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_volatility

    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, 1) for _ in range(n_assets)]

    result = minimize(sharpe_ratio, init_weights, bounds=bounds, constraints=constraints)
    return result.x

def efficient_frontier(expected_returns, cov_matrix, num_points=100):
    risks, returns = [], []
    for target_return in np.linspace(min(expected_returns), max(expected_returns), num_points):
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return},
        ]
        bounds = [(0, 1) for _ in range(len(expected_returns))]
        result = minimize(
            lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
            np.ones(len(expected_returns)) / len(expected_returns),
            bounds=bounds,
            constraints=constraints
        )
        if result.success:
            risks.append(result.fun)
            returns.append(target_return)
    return risks, returns

# -------------------------------------------------
# MAIN LOGIC WITH STREAMLIT INTERFACE
# -------------------------------------------------
def main():
    st.title("Cryptocurrency Portfolio Optimizer")

    # Streamlit UI for user input
    crypto_pairs = st.multiselect(
        "Select Cryptocurrencies:",
        ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT"],
        default=["BTCUSDT", "ETHUSDT"]
    )

    interval = "1d"

    horizon = st.slider("Prediction Horizon (days):", min_value=1, max_value=30, value=7)

    # Set start_date and end_date dynamically
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    if st.button("Optimize Portfolio"):
        with st.spinner("Fetching data and optimizing portfolio..."):
            # Fetch historical data
            data_dict = fetch_data(crypto_pairs, interval, start_date, end_date)

            # Calculate expected returns (using LSTM prediction) and volatilities (using GARCH)
            expected_returns = []
            for pair, data in data_dict.items():
                scaled_data = function2.apply_minmax_scaler(data)
                X, y = function2.generate_sequences(scaled_data, seq_length=10)
                prediction = function2.recursive_lstm_prediction(X, y, horizon)
                returns = (prediction[-1] - prediction[0]) / prediction[0] if len(prediction) > 1 else 0
                expected_returns.append(returns)
            expected_returns = np.array(expected_returns)

            vol_array = calculate_volatility(data_dict, interval)

            # Construct covariance matrix
            cov_matrix = covariance_matrix(vol_array)

            # Optimize portfolio for maximum Sharpe Ratio
            optimal_weights = optimize_portfolio(expected_returns, cov_matrix)

            # Display results
            st.subheader("Optimal Portfolio Weights")
            for crypto, weight in zip(crypto_pairs, optimal_weights):
                st.write(f"{crypto}: {weight:.2%}")

            # Generate efficient frontier
            risks, returns = efficient_frontier(expected_returns, cov_matrix)

            # Plot efficient frontier
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(risks, returns, label="Efficient Frontier", color='blue')
            optimal_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
            optimal_return = np.dot(optimal_weights, expected_returns)
            ax.scatter(optimal_risk, optimal_return, color="red", label="Optimal Portfolio", zorder=5)
            ax.text(optimal_risk, optimal_return, "Optimal Portfolio", fontsize=10, ha='right')

            # Labels and legend
            ax.set_xlabel("Risk (Standard Deviation)")
            ax.set_ylabel("Expected Return")
            ax.set_title("Efficient Frontier and Optimal Portfolio")
            ax.legend()
            ax.grid()

            st.pyplot(fig)

if __name__ == "__main__":
    main()
