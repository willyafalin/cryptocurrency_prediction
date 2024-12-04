
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Charger les données depuis le fichier CSV
dataframe = pd.read_csv("BTCUSDT_hourly_2023.csv")

# Conversion de la colonne 'timestamp' en format datetime
dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])

# Utiliser la colonne 'timestamp' comme index
dataframe.set_index('timestamp', inplace=True)

# Garder uniquement la colonne 'close' pour le modèle LSTM
close_prices = dataframe['close']

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


print(dataframe.head())
print(dataframe['close'].head())

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

print(train_predict)
print(test_predict)

plt.show(block=True)
