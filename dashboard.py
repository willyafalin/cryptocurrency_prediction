import streamlit as st 
import time
from PIL import Image
import function
from datetime import datetime
import pandas as pd 
from binance.client import Client

from datetime import datetime

# Obtenir la date actuelle
time_now = datetime.now()

# Format de la date attendu par la fonction get_historical_data
format_date = "%d %b, %Y"

# Formater la date actuelle
date = time_now.strftime(format_date)

# Modifier l'année pour obtenir la date de l'année précédente
date_ = date[:-4] + str(int(date[-4:]) - 1)

# Définir les dates de début et de fin
start_date = date_
end_date = date


#image = Image.open('app_image.jpeg')
#st.image(image, width = 700)
#st.info('Use of Long short-terme memory deep learning model to predict the close price of a cryptocurrency', icon="ℹ️")
#st.warning('Warning : the given prediction is by no means a reliable investment !')

crypto_selectbox = st.selectbox('Choose your Cryptocurrency', 
                             ('BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'TRXUSDT', 'USDCUSDT'))


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
if button_prediction:
    with st.spinner('Prediction in progress ...'):
        # Obtenir les données historiques
        interval = Client.KLINE_INTERVAL_1HOUR if select_time == 'Hourly' else Client.KLINE_INTERVAL_1DAY
        data = function.get_historical_data(crypto_selectbox, interval, start_date, end_date)

        # Mise à l'échelle et prédiction des prix
        scaled_data = function.scaling_data(data)
        X, y = function.create_sequences(scaled_data, sequence_length=10)
        prediction = function.recursive_prediction(X, y, t)
        prediction = function.inverse_scalling(prediction, data)
        df = pd.DataFrame(prediction[-t:], columns=['Price USD'])

        # Déterminer les paramètres GARCH selon l'intervalle
        if select_time == 'Hourly':
            p, q, dist = 2, 2, 't'  # Modèle GARCH(2,2) avec distribution t-student
        else:
            p, q, dist = 1, 1, 'normal'  # Modèle GARCH(1,1) avec distribution normale

        # Analyse GARCH pour la volatilité
        garch_data = function.garch_analysis_for_dashboard(data, p=p, q=q, dist=dist, last_n_days=t)
        df['Volatility'] = garch_data['predicted_volatility_scaled'].values

        st.success('Done', icon="✅")

        # Visualisation combinée des prix et des bandes de volatilité
        import plotly.graph_objects as go

        fig = go.Figure()

        # Ajouter les prix prédits
        fig.add_trace(go.Scatter(x=df.index, y=df['Price USD'], mode='lines', name='Predicted Price'))

        # Ajouter les bandes de volatilité
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Price USD'] + df['Volatility'],
            mode='lines',
            line=dict(width=0),
            name='Upper Bound',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Price USD'] - df['Volatility'],
            mode='lines',
            fill='tonexty',  # Remplir la zone entre les bandes
            line=dict(width=0),
            name='Lower Bound',
            showlegend=False
        ))

        # Mise à jour des paramètres du graphique
        fig.update_layout(
            title="Predicted Price with Volatility Bands",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_white"
        )

        # Affichage du graphique
        st.plotly_chart(fig)


