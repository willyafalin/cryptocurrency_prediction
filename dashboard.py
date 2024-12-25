import streamlit as st
import time
from PIL import Image
import function  # On suppose que garch_multi_forecast est dedans
from datetime import datetime
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime

# Plotly
import plotly.graph_objects as go

# ------------------------------------------------------------------
# CURRENT DATE
time_now = datetime.now()
format_date = "%d %b, %Y"
date = time_now.strftime(format_date)

# One year ago (simple string manipulation)
date_ = date[:-4] + str(int(date[-4:]) - 1)

# Start/end dates
start_date = date_
end_date = date

# ------------------------------------------------------------------
# STREAMLIT UI

crypto_selectbox = st.selectbox(
    'Choose your Cryptocurrency', 
    ('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 
     'ADAUSDT', 'DOGEUSDT', 'TRXUSDT', 'USDCUSDT')
)

t = None
select_time = st.radio('Choose the Data interval', ['Hourly', 'Daily'])

if select_time == 'Hourly':
    slider_hour = st.slider('Choose the time in hour to predict', 1, 50, 1)
    t = slider_hour
else:
    select_days = st.slider('Choose the time in days to predict', 1, 10, 1)
    t = select_days

button_prediction = st.button('Submit data')

interval = None
data = None
scaled_data = None

# ------------------------------------------------------------------
# MAIN LOGIC

if button_prediction:
    with st.spinner('Prediction in progress ...'):
        
        # 1) Fetch historical data
        interval = (
            Client.KLINE_INTERVAL_1HOUR 
            if select_time == 'Hourly' 
            else Client.KLINE_INTERVAL_1DAY
        )
        data = function.get_historical_data(
            crypto_selectbox, interval, start_date, end_date
        )

        # 2) LSTM PART: Scale the data & build sequences
        scaled_data = function.scaling_data(data)
        X, y = function.create_sequences(scaled_data, sequence_length=10)
        
        # 3) LSTM recursive prediction
        prediction = function.recursive_prediction(X, y, t)
        
        # 4) Inverse scaling back to original price units
        prediction = function.inverse_scalling(prediction, data)
        
        # Les t derniers points prédits (futurs)
        df = pd.DataFrame(prediction[-t:], columns=['Price USD'])
        df.reset_index(drop=True, inplace=True)  
        # => index 0..(t-1), 
        # la ligne 0 correspond à la prévision de (T+1),
        # la ligne 1 -> (T+2), etc.

        # ------------------------------------------------------------------
        # 5) GARCH PART: Paramètres GARCH
        if select_time == 'Hourly':
            p, q, dist = 2, 2, 't'      # GARCH(2,2) with t-student
        else:
            p, q, dist = 1, 1, 'normal' # GARCH(1,1) with normal dist

        # On appelle la NOUVELLE FONCTION multi-step GARCH
        # qu'on suppose définie dans "function.py"
        # garch_multi_forecast(data, p, q, dist='normal', horizon=5, window_realized=10)
        horizon = t
        garch_result = function.garch_multi_forecast(
            data=data,
            p=p,
            q=q,
            dist=dist,
            horizon=horizon,
            window_realized=10
        )
        
        # garch_result contient:
        #   - 'predicted_volatility_scaled_1' ... '_2' ... jusqu'à '_t'
        #   - 'predicted_volatility_original_1' ... etc.
        # On veut associer le pas i (ligne i dans df) avec 'predicted_volatility_scaled_i'.
        
        # Pour simplifier, on va créer une nouvelle colonne 'Volatility' dans df 
        # => la "volatilité" correspond au i-ème step. 
        # i varie de 1 à t. 
        # Sur la ligne 0 du df => i=1 => on prend 'predicted_volatility_original_1'
        # Sur la ligne 1 => i=2 => on prend 'predicted_volatility_original_2'
        # etc.

        # On crée un array vide
        vol_array = np.zeros(t)
        for i in range(t):
            col_name = f'predicted_volatility_original_{i+1}'
            if col_name in garch_result.columns:
                # Récupérer la dernière valeur non-NaN (index T+horizon)
                vals = garch_result[col_name].dropna()
                if len(vals) > 0:
                    # On prend la dernière en date
                    vol_array[i] = vals.iloc[-1]
                else:
                    vol_array[i] = 0.0
            else:
                vol_array[i] = 0.0
        
        # Maintenant, vol_array[i] correspond à la volatilité prévue au step i+1
        df['Volatility'] = vol_array

        st.success('Done', icon="✅")

        # ------------------------------------------------------------------
        # -- IMPROVEMENT: Log-return-based approach for bands --

        mu = 0.0   # Suppose drift = 0
        k = 1.0    # 1-sigma band

        # Price(t+1) ~ Price(t) * exp( mu ± k * sigma )
        df['UpperBand'] = df['Price USD'] * np.exp(mu + k * df['Volatility'])
        df['LowerBand'] = df['Price USD'] * np.exp(mu - k * df['Volatility'])

        # ------------------------------------------------------------------
        # PLOTLY CHART
        fig = go.Figure()

        # 1) Predicted Price
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Price USD'], 
            mode='lines', 
            name='Predicted Price'
        ))

        # 2) Upper Band
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['UpperBand'],
            mode='lines',
            line=dict(width=0),
            name='Upper Band',
            showlegend=False
        ))

        # 3) Lower Band + fill
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['LowerBand'],
            mode='lines',
            fill='tonexty',
            line=dict(width=0),
            name='Lower Band',
            showlegend=False
        ))

        fig.update_layout(
            title="Predicted Price with Multi-step GARCH Volatility Bands",
            xaxis_title="Future Steps (index 0..t-1)",
            yaxis_title="Price (USD)",
            template="plotly_white"
        )

        st.plotly_chart(fig)
