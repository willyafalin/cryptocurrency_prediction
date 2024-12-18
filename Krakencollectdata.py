
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



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
dataframe = pd.read_csv("BTCUSD_kraken_hourly_2019_to_2024.csv")

# Conversion de la colonne 'time' en format datetime
dataframe['time'] = pd.to_datetime(dataframe['time'])

# Variables indépendantes (features) et dépendante (target)
X = dataframe[['open_price', 'high_price', 'low_price', 'trade_volume']]
y = dataframe['close_price']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Régression linéaire simple (utilisant uniquement 'open_price')
X_simple_train = X_train[['open_price']]
X_simple_test = X_test[['open_price']]

simple_model = LinearRegression()
simple_model.fit(X_simple_train, y_train)

# Prédiction
y_simple_pred = simple_model.predict(X_simple_test)

# Régression linéaire multiple
multiple_model = LinearRegression()
multiple_model.fit(X_train, y_train)

# Prédiction
y_multiple_pred = multiple_model.predict(X_test)

# Évaluation des modèles
simple_mse = mean_squared_error(y_test, y_simple_pred)
simple_r2 = r2_score(y_test, y_simple_pred)

multiple_mse = mean_squared_error(y_test, y_multiple_pred)
multiple_r2 = r2_score(y_test, y_multiple_pred)

print("Régression Linéaire Simple :")
print("MSE:", simple_mse)
print("R²:", simple_r2)

print("\nRégression Linéaire Multiple :")
print("MSE:", multiple_mse)
print("R²:", multiple_r2)

# Visualisation des résultats de la régression linéaire simple
plt.scatter(X_simple_test, y_test, color='blue', label='Données Réelles')
plt.plot(X_simple_test, y_simple_pred, color='red', label='Prédictions')
plt.xlabel('Prix d\'ouverture')
plt.ylabel('Prix de clôture')
plt.title('Régression Linéaire Simple')
plt.legend()
plt.show()

# Visualisation des résultats de la régression linéaire multiple
plt.scatter(y_test, y_multiple_pred, color='green', label='Prédictions vs Réel')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel('Prix de clôture réel')
plt.ylabel('Prix de clôture prédit')
plt.title('Régression Linéaire Multiple')
plt.legend()
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
dataframe = pd.read_csv("BTCUSD_kraken_hourly_2019_to_2024.csv")

# Conversion de la colonne 'time' en format datetime
dataframe['time'] = pd.to_datetime(dataframe['time'])

# Calcul de la matrice de corrélation
correlation_matrix = dataframe[['open_price', 'high_price', 'low_price', 'close_price', 'trade_volume']].corr()

# Affichage de la matrice de corrélation
print(correlation_matrix)

# Visualisation de la matrice de corrélation avec une heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True)
plt.title('Heatmap de la Matrice de Corrélation')
plt.show()




import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt

# Charger les données depuis le fichier CSV
dataframe = pd.read_csv("BTCUSD_kraken_hourly_2019_to_2024.csv")

# Conversion de la colonne 'time' en format datetime
dataframe['time'] = pd.to_datetime(dataframe['time'])

# Utiliser la colonne 'time' comme index
dataframe.set_index('time', inplace=True)

# Garder uniquement la colonne 'close_price' pour le modèle ARIMA
close_prices = dataframe['close_price']

# Trouver les meilleurs paramètres ARIMA automatiquement
model = pm.auto_arima(close_prices, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)

# Résumer le modèle
print(model.summary())

# Faire des prédictions
n_periods = 24  # Par exemple, prédire les 24 prochaines heures
forecast, conf_int = model.predict(n_periods=n_periods, return_conf_int=True)

# Convertir les prévisions en série temporelle
forecast_index = pd.date_range(start=close_prices.index[-1], periods=n_periods+1, freq='H')[1:]
forecast_series = pd.Series(forecast, index=forecast_index)

# Visualisation des résultats
plt.figure(figsize=(12, 6))
plt.plot(close_prices, label='Données Réelles')
plt.plot(forecast_series, label='Prédictions', color='red')
plt.fill_between(forecast_series.index, conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Prix de clôture')
plt.title('Prédictions des prix du Bitcoin avec ARIMA (Optimisation Automatique)')
plt.legend()
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Charger les données depuis le fichier CSV
dataframe = pd.read_csv("BTCUSD_kraken_hourly_2019_to_2024.csv")

# Conversion de la colonne 'time' en format datetime
dataframe['time'] = pd.to_datetime(dataframe['time'])

# Utiliser la colonne 'time' comme index
dataframe.set_index('time', inplace=True)

# Garder uniquement la colonne 'close_price' pour le modèle LSTM
close_prices = dataframe['close_price']

# Normaliser les prix de clôture
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices.values.reshape(-1, 1))

# Diviser les données en ensembles d'entraînement et de test
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Fonction pour créer un ensemble de données pour l'entraînement LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape des données pour LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Création du modèle LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Prédictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Reconvertir les données prédictes à l'échelle originale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Visualisation des résultats
plt.figure(figsize=(14, 7))
plt.plot(close_prices.index, close_prices, label='Données Réelles')
train_predict_plot = np.empty_like(scaled_data)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step, :] = train_predict
plt.plot(close_prices.index, train_predict_plot, label='Prédictions Entraînement', color='orange')

test_predict_plot = np.empty_like(scaled_data)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict
plt.plot(close_prices.index, test_predict_plot, label='Prédictions Test', color='red')

plt.xlabel('Date')
plt.ylabel('Prix de clôture')
plt.title('Prédiction des Prix du Bitcoin avec LSTM')
plt.legend()
plt.show()


