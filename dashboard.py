# -------------------------------------------------
# IMPORTS
# -------------------------------------------------
import streamlit as st
import time
from PIL import Image
import function2 
from datetime import datetime
import pandas as pd
import numpy as np
from binance.client import Client

# For visualization
import plotly.graph_objects as go


# -------------------------------------------------
# CURRENT DATE CONFIGURATION
# -------------------------------------------------
# Get the current date and format it
time_now = datetime.now()
format_date = "%d %b, %Y"
date = time_now.strftime(format_date)

# Calculate the date one year ago
date_ = date[:-4] + str(int(date[-4:]) - 1)

# Define start and end dates
start_date = date_
end_date = date


# -------------------------------------------------
# STREAMLIT USER INTERFACE
# -------------------------------------------------
# Dropdown to select the cryptocurrency pair
crypto_selectbox = st.selectbox(
    'Choose your Cryptocurrency',
    ('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
     'ADAUSDT', 'DOGEUSDT', 'TRXUSDT', 'USDCUSDT')
)

# Radio buttons and sliders for time interval selection
t = None
select_time = st.radio('Choose the Data interval', ['Hourly', 'Daily'])

if select_time == 'Hourly':
    slider_hour = st.slider('Choose the time in hour to predict', 1, 50, 1)
    t = slider_hour
else:
    select_days = st.slider('Choose the time in days to predict', 1, 10, 1)
    t = select_days

# Button to trigger predictions
button_prediction = st.button('Submit data')


# -------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------
if button_prediction:
    with st.spinner('Prediction in progress ...'):

        # -------------------------------------------------
        # FETCH HISTORICAL DATA
        # -------------------------------------------------
        # Determine the interval (hourly or daily)
        interval = (
            Client.KLINE_INTERVAL_1HOUR
            if select_time == 'Hourly'
            else Client.KLINE_INTERVAL_1DAY
        )
        # Retrieve historical data using the custom function
        data = function2.fetch_binance_data_full(
            crypto_selectbox, interval, start_date, end_date
        )

        # -------------------------------------------------
        # LSTM PART: SCALE DATA AND PREDICT PRICES
        # -------------------------------------------------
        # Scale the data and create sequences for LSTM
        scaled_data = function2.apply_minmax_scaler(data)
        X, y = function2.generate_sequences(scaled_data, seq_length=10)

        # Perform recursive predictions with LSTM
        prediction = function2.recursive_lstm_prediction(X, y, t)

        # Convert predictions back to the original scale
        prediction = function2.reverse_scaling(prediction, data)

        # Store predictions in a DataFrame
        df = pd.DataFrame(prediction[-t:], columns=['Price USD'])
        df.reset_index(drop=True, inplace=True)

        # -------------------------------------------------
        # GARCH PART: CALCULATE VOLATILITY
        # -------------------------------------------------
        # Define GARCH parameters based on time interval
        if select_time == 'Hourly':
            p, q, dist = 2, 2, 't'  # GARCH(2,2) with t-distribution
        else:
            p, q, dist = 1, 1, 'normal'  # GARCH(1,1) with normal distribution

        # Call the multi-step GARCH function
        garch_result = function2.garch_multi_forecast(
            data=data,
            p=p,
            q=q,
            dist=dist,
            horizon=t,
            window_realized=10
        )

        # Extract predicted volatility for each step
        vol_array = np.zeros(t)
        for i in range(t):
            col_name = f'predicted_volatility_original_{i+1}'
            if col_name in garch_result.columns:
                # Retrieve the last non-NaN value for this step
                vals = garch_result[col_name].dropna()
                if len(vals) > 0:
                    vol_array[i] = vals.iloc[-1]
                else:
                    vol_array[i] = 0.0
            else:
                vol_array[i] = 0.0

        # Add the volatility to the DataFrame
        df['Volatility'] = vol_array

        st.success('Done', icon="✅")

        # -------------------------------------------------
        # CALCULATE CONFIDENCE BANDS
        # -------------------------------------------------
        mu = 0.0   # Assume no drift
        k = 1.0    # 1-sigma confidence level

        # Calculate upper and lower bounds using log-returns
        df['UpperBand'] = df['Price USD'] * np.exp(mu + k * df['Volatility'])
        df['LowerBand'] = df['Price USD'] * np.exp(mu - k * df['Volatility'])

        # -------------------------------------------------
        # PLOTLY VISUALIZATION
        # -------------------------------------------------
        fig = go.Figure()

        # Add the predicted price line
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Price USD'],
            mode='lines',
            name='Predicted Price'
        ))

        # Add the upper and lower bounds
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['UpperBand'],
            mode='lines',
            line=dict(width=0),
            name='Upper Band',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['LowerBand'],
            mode='lines',
            fill='tonexty',  # Fill the area between bounds
            line=dict(width=0),
            name='Lower Band',
            showlegend=False
        ))

        # Customize the layout
        fig.update_layout(
            title="Predicted Price with Multi-step GARCH Volatility Bands",
            xaxis_title="Future Steps (index 0..t-1)",
            yaxis_title="Price (USD)",
            template="plotly_white"
        )

        # Display the chart
        st.plotly_chart(fig)

