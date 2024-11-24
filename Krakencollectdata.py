import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib
import time

# Fonction pour récupérer les données OHLC depuis Kraken
def fetch_ohlc_data(symbol, time_frame, start_time, end_time=None):
    all_data = []
    current_time = start_time

    while True:
        endpoint = 'https://api.kraken.com/0/public/OHLC'
        parameters = {
            'pair': symbol,
            'interval': time_frame,
            'since': current_time
        }
        reply = requests.get(endpoint, params=parameters)
        json_data = reply.json()

        if reply.status_code == 200:
            ohlc_values = json_data['result'][symbol]
            dataframe = pd.DataFrame(ohlc_values, columns=[
                'time', 'open_price', 'high_price', 'low_price', 'close_price', 'end_time', 'trade_volume',
                'quote_volume', 'trade_count', 'buy_volume', 'buy_quote_volume', 'ignore'
            ])
            all_data.append(dataframe)
            current_time = int(dataframe['time'].iloc[-1])  # Dernier timestamp récupéré
            time.sleep(1)  # Pause de 1 seconde entre les requêtes

            # Arrêt si le timestamp actuel dépasse la période spécifiée
            if end_time and current_time >= end_time:
                break
        else:
            print(f"Erreur lors de la récupération des données de Kraken : {reply.status_code}")
            break

    full_data = pd.concat(all_data)
    full_data['time'] = pd.to_datetime(full_data['time'], unit='s')
    full_data[['open_price', 'high_price', 'low_price', 'close_price', 'trade_volume']] = full_data[['open_price', 'high_price', 'low_price', 'close_price', 'trade_volume']].astype(float)
    return full_data

# Configuration des paramètres
symbol = 'XXBTZUSD'
time_frame = 60  # Intervalle de  heure
start_time = int(pd.Timestamp('2019-01-01').timestamp())  # À partir du 1er janvier 2019
end_time = int(pd.Timestamp('2024-11-24').timestamp())  # Jusqu'à aujourd'hui

# Récupération des données avec plusieurs requêtes successives
dataframe = fetch_ohlc_data(symbol, time_frame, start_time, end_time)

if dataframe is not None:
    # Enregistrement des données dans un fichier CSV
    file_name = "BTCUSD_kraken_hourly_2019_to_2024.csv"
    dataframe.to_csv(file_name, index=False)
    print(f"Données enregistrées dans {file_name}")

    # Chargement des données depuis le fichier CSV
    csv_file = "BTCUSD_kraken_hourly_2019_to_2024.csv"
    dataframe = pd.read_csv(csv_file)

    # Conversion de la colonne 'time' en format datetime
    dataframe['time'] = pd.to_datetime(dataframe['time'])

    # Vérification des données
    print(dataframe.head())
    print(dataframe.info())

    # Calcul du VWAP (Volume Weighted Average Price)
    dataframe['cum_volume'] = dataframe['trade_volume'].cumsum()
    dataframe['cum_vwap'] = (dataframe['close_price'] * dataframe['trade_volume']).cumsum()
    dataframe['VWAP'] = dataframe['cum_vwap'] / dataframe['cum_volume']

    # Calcul de la volatilité (écart-type des rendements logarithmiques)
    dataframe['log_return'] = dataframe['close_price'].apply(lambda x: np.log(x)).diff()
    dataframe['volatility'] = dataframe['log_return'].rolling(window=30).std() * np.sqrt(30)  # Volatilité sur 30 périodes

    # Calcul des Bandes de Bollinger
    dataframe['upper_band'], dataframe['middle_band'], dataframe['lower_band'] = talib.BBANDS(dataframe['close_price'], timeperiod=20)

    # Calcul de la Volatilité de Chaikin (Chaikin Volatility)
    dataframe['high_low_diff'] = dataframe['high_price'] - dataframe['low_price']
    dataframe['chaikin_volatility'] = talib.SMA(dataframe['high_low_diff'], timeperiod=10) / talib.SMA(dataframe['high_low_diff'], timeperiod=10).shift(10) - 1

    # Affichage des 100 premières lignes
    print("Affichage des 100 premières lignes de données :")
    print(dataframe.head(100))

    # Graphique combiné des prix de clôture et des indicateurs
    plt.figure(figsize=(14, 8))
    plt.plot(dataframe['time'], dataframe['close_price'], label='Prix de clôture')
    plt.plot(dataframe['time'], dataframe['VWAP'], label='VWAP', linestyle='--')
    plt.plot(dataframe['time'], dataframe['upper_band'], label='Upper Band', linestyle='--')
    plt.plot(dataframe['time'], dataframe['middle_band'], label='Middle Band', linestyle='--')
    plt.plot(dataframe['time'], dataframe['lower_band'], label='Lower Band', linestyle='--')
    plt.plot(dataframe['time'], dataframe['chaikin_volatility'], label='Chaikin Volatility', linestyle=':')
    plt.title("Prix de clôture BTC/USD avec VWAP, Bandes de Bollinger et Volatilité de Chaikin (2019-2024)")
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid()
    plt.show()

    # Graphique des volumes
    plt.figure(figsize=(12, 6))
    plt.bar(dataframe['time'], dataframe['trade_volume'], width=0.01)
    plt.title("Volumes échangés BTC/USD (2019-2024)")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.grid()
    plt.show()

    # Graphique de la volatilité
    plt.figure(figsize=(12, 6))
    plt.plot(dataframe['time'], dataframe['volatility'], label='Volatilité')
    plt.title("Volatilité BTC/USD (2019-2024)")
    plt.xlabel("Date")
    plt.ylabel("Volatilité (écart-type des rendements)")
    plt.legend()
    plt.grid()
    plt.show()

    # Graphique des Bandes de Bollinger
    plt.figure(figsize=(14, 8))
    plt.plot(dataframe['time'], dataframe['close_price'], label='Prix de clôture')
    plt.plot(dataframe['time'], dataframe['upper_band'], label='Bande Supérieure', linestyle='--')
    plt.plot(dataframe['time'], dataframe['middle_band'], label='Bande Centrale', linestyle='--')
    plt.plot(dataframe['time'], dataframe['lower_band'], label='Bande Inférieure', linestyle='--')
    plt.title("Bandes de Bollinger BTC/USD (2019-2024)")
    plt.xlabel("Date")
    plt.ylabel("Prix (USD)")
    plt.legend()
    plt.grid()
    plt.show()

    # Graphique de la Volatilité de Chaikin
    plt.figure(figsize=(14, 8))
    plt.plot(dataframe['time'], dataframe['chaikin_volatility'], label='Volatilité de Chaikin')
    plt.title("Volatilité de Chaikin BTC/USD (2019-2024)")
    plt.xlabel("Date")
    plt.ylabel("Chaikin Volatility")
    plt.legend()
    plt.grid()
    plt.show()

else:
    print("Impossible de récupérer les données de Kraken.")
