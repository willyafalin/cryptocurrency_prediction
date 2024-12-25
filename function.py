import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import requests 
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels
import datetime
from binance.client import Client
import time
from arch import arch_model
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# Fonction pour extraire des données historiques de Binance
def get_historical_data(symbol, interval, start_date, end_date):
    
    client = Client()
    
    # Convertir les dates en timestamps
    start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
    klines = []

    # Télécharger les données par lots (max 1000 chandeliers par requête)
    while start_ts < end_ts:
        temp_klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=1000,
            startTime=start_ts
        )
        if not temp_klines:
            break
        klines.extend(temp_klines)
        start_ts = temp_klines[-1][0] + 1  # Passer au prochain lot
        time.sleep(0.1)  # Respecter les limites de l'API

    # Transformer les données en DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convertir en format lisible
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


#Graphic fonction
def visualize_data(data, symbol, interval):
    """

    Aim : Visualize the close cotation of  the considered Cryptocurrencie on the considered interval
    
    Input : 
    - data :
        type: pandas DataFrame 
                
    - symbol :
        type: string
        for instance 'BTC/USD'
        
    - interval : 
        type: string
        for instance '1day' or '5min'
        
    Output : Plot

    """
 
    sns.set(style = 'darkgrid')
    plt.title('Close cotation of '+ str(symbol))
    plt.xlabel(str(interval))
    plt.ylabel('USD')
    sns.lineplot(x = data.index, y = data['close'], color = 'green')
       
    plt.show()


def finance_visualize(data, symbol, interval):

    """

    Aim : Visualize the candle of the market of the considered Cryptocurrencie
    on the considered interval
    
    Input : 
    - data :
        type: pandas DataFrame 
                
    - symbol :
        type: string
        for instance 'BTC/USD'
        
    - interval : 
        type: string
        for instance '1day' or '5min'
        
    Output : Plot
    
    """

    # Creating a candlestick chart using Plotly.
    fig = go.Figure(data=[go.Candlestick(x=data['timestamp'],
                                     open=data['open'],
                                     high=data['high'],
                                     low=data['low'],
                                     close=data['close'])])
    
    fig.update_layout(title='Cotation of ' + str(symbol) + ' per ' + str(interval) ,
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  xaxis_rangeslider_visible=False)



    fig.show()



def visualize_with_indicator(data, symbol, interval, indicator):

    """

    Aim : Visualize the candle of the market of the considered Cryptocurrencie
    on the considered interval
    
    Input : 
    - data :
        type: pandas DataFrame 
                
    - symbol :
        type: string
        for instance 'BTC/USD'
        
    - interval : 
        type: string
        for instance '1day' or '5min'
    - indicator :
        type: string
        RSI EMA ATR
        
    Output : Plot
    


    """
 
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Candles', str(indicator)])
    candle = go.Candlestick(x=data['timestamp'],
                                        open=data['open'],
                                        high=data['high'],
                                        low=data['low'],
                                        close=data['close'])

    fig.add_trace(candle, row=1, col =1)


    rsi_trace=go.Scatter(x=data['timestamp'], y=data[str(indicator)], mode='lines', name=str(indicator), yaxis='y2')

    fig.add_trace(rsi_trace, row=2, col=1)

    
    fig.update_layout(title='Cotation of ' + str(symbol) + ' per ' + str(interval) ,
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  xaxis_rangeslider_visible=False)



    fig.show()


def add_indicators(data, period=14):

    """
    input :
    data : type pandas DataFrame 

    output : a pandas DataFrame wich contain the three indicators 
    
    """
    
    ema = ta.trend.ema_indicator(close = data['close'], window = period).dropna()
    rsi = ta.momentum.rsi(close=data['close'], window=period).dropna()
    atr = ta.volatility.AverageTrueRange(close=data['close'],high=data['high'], low=data['low'], window=period).average_true_range()
    atr = atr[atr>0]
    data = pd.DataFrame(data.loc[period-1:])


    data['RSI'] = rsi
    data['EMA'] = ema
    data['ATR'] = atr



    return data.reset_index().drop('index', axis=1)




def scaling_data(data):

    """

    aim : Scale the data frma with MinMax scaler to apply machine learning models 

    input : 
    data : type pandas DataFrame

    output : scaled DataFrame

    """



    
    # Create a copy of the data to avoid modifying the original DataFrame
    scaled_data = data.copy()

    # Find the global minimum and maximum values from the 'low' and 'high' columns, respectively
    global_min = data['low'].min()
    global_max = data['high'].max()

    # Define a function to scale each value
    def scale_value(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    # Scale the 'open' and 'close' prices with respect to the global min and max
    scaled_data['open'] = data['open'].apply(scale_value, args=(global_min, global_max))
    scaled_data['close'] = data['close'].apply(scale_value, args=(global_min, global_max))

    # we directly scale 'high' and 'low' with global min and max as well
    scaled_data['high'] = data['high'].apply(scale_value, args=(global_min, global_max))
    scaled_data['low'] = data['low'].apply(scale_value, args=(global_min, global_max))

    # Scale other columns individually 
    for column in data.columns:
        if column not in ['timestamp', 'open', 'high', 'low', 'close']:
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Scale each column separately
            scaled_column = scaler.fit_transform(data[[column]])
            # Store the scaled values back in the DataFrame
            scaled_data[column] = scaled_column.flatten()

    return scaled_data


def data_preprocess(scaled_data,regressor, prediction_time):

    """  
    Aim : Shift the data to do regression on time series 
    for example, if you want to predict the next 30 days by changing the data, 
    each line of price is associated with the value taken 30 days later.

    Input:
    - scaled_data   type : DataFrame
    - prediction_time   type : int

    Output:w
    - price Type : Numpy Array 
    - target Type : Numpy Array
    
    """
   
    target = scaled_data['close'].shift(-prediction_time).dropna()
    target = np.array(target).reshape(-1, 1)

    price = np.array(scaled_data[regressor])[:-prediction_time]
    
    return price, target 


def visualize_model(prediction_matrix,scaled_data, zoom = None):

    plt.xlabel('Days')
    plt.ylabel('BTC/USD ($)(scaled data)')
    plt.plot(scaled_data['close'])
    plt.plot(prediction_matrix[['close', 'prediction']])
    plt.legend(['Real Price', 'Real, price', 'Prediction'])
    if zoom is not None : 
        plt.xlim(zoom[0], zoom[1])
    plt.title('Prediction of close price of BTC/USD for the Last Month by Linear Regression')
    plt.show


def apply_linear_regression(scaled_data, prediction_time, price, target, regressor):

    """

    Input : 
    scaled_data : type pandas DataFrame wich contains the scaled data 
    prediction_time : type int, if prediction_time = 5 we predict the 5th next value 
    price : type numpy array
    target : numpy array 
    regressor : type list of strings 

    output : 
    prediction_matrix : type pandas DataFrame wich containts the realised price ans the predictions 
    future : type numpy array which contains the prediction_time future parice form the last date of the data set 
    r2 : type float it's the well known R2 of the linear regression

    """



    price_train, price_test, target_train, target_test = train_test_split(price, target, test_size = 0.3) # we split the data into test and training sets
    lr = LinearRegression().fit(price_train, target_train) # we fit the model on the training set 

    price_to_predict = price[-prediction_time:] # we take the last slice to predict 
    lr_prediction = lr.predict(price_to_predict) # we apply the model 


    prediction_matrix = pd.DataFrame(scaled_data['close'].tail(prediction_time)) # creat a DataFrame which contains the realised prices 
    prediction_matrix['prediction'] = lr_prediction # we add the prediction as a new column in order to compare with a plot 

    price_to_future = np.array(scaled_data[regressor])[-prediction_time:] # we take the last slice of our data to predict future values 
    future = lr.predict(price_to_future)

    target_predict = lr.predict(price_test)
    r2 = r2_score(target_test, target_predict) # we compute the R2

    return prediction_matrix, future, r2 


def visualize_future(scaled_data, future, zoom = None):

    plt.xlabel('Days')
    plt.ylabel('BTC/USD ($)(scaled data)')
    arr1 = np.array(scaled_data['close']).reshape(-1, 1)
    arr2 = np.array(future).reshape(-1, 1) 
    ct = np.concatenate((arr1, arr2))
    plt.axvline(x = arr1.shape[0], color = 'r', linestyle = '--', label = 'Prediction')
    plt.plot(ct)
    if zoom is not None : 
        plt.xlim(zoom[0], zoom[1])
    plt.title('Prediction of close price of BTC/USD')

    plt.show


def apply_svr(scaled_data, prediction_time, price, target, regressor, best_C, best_gamma):

    """

    Input : 
    scaled_data : type pandas DataFrame wich contains the scaled data 
    prediction_time : type int, if prediction_time = 5 we predict the 5th next value 
    price : type numpy array
    target : numpy array 
    regressor : type list of strings 
    best_C and best_gamma : type float wich are parameters of the SVR model 

    output : 
    prediction_matrix : type pandas DataFrame wich containts the realised price ans the predictions 
    future : type numpy array which contains the prediction_time future parice form the last date of the data set 
    svr_accuracy : type float it measures the accuracy or how well the model fit the data
    
    """


   

    price_train, price_test, target_train, target_test = train_test_split(price, target, test_size = 0.3)
    svr_rbf = SVR(kernel = 'rbf', C = best_C, gamma= best_gamma)
    svr_rbf.fit(price_train, np.ravel(target_train))

    price_to_predict = price[-prediction_time:] 
    svr_prediction = svr_rbf.predict(price_to_predict)


    prediction_matrix = pd.DataFrame(scaled_data['close'].tail(prediction_time))
    prediction_matrix['prediction'] = svr_prediction

    price_to_future = np.array(scaled_data[regressor])[-prediction_time:]
    future = svr_rbf.predict(price_to_future)

    svr_accuracy = svr_rbf.score(price_test, target_test)

    return prediction_matrix, future, svr_accuracy 


def cross_validation_parameter(param_grid, price_train, target_train):

    """
    
    Aim : Find the best parameters C and Gamma by Cross Validation 

    Input : 
    param_grid : type dictioary which contains the values of gamme and C we want to test 
    price_train and target_train : numpy array

    output : Gives the Best C and the Best Gamma among the value of param_grid
    
    """

    svr_rbf = SVR(kernel = 'rbf') # We use the Support Vector Regression
    search = GridSearchCV(svr_rbf, param_grid, cv=3, scoring = 'neg_mean_squared_error', n_jobs=-1)
    search.fit(price_train, np.ravel(target_train)) # we fit the cross validation on the data
    
    best_C = search.best_params_['C']
    best_gamma = search.best_params_['gamma']

    return best_C, best_gamma




# Function to create sequences for training the model
def create_sequences(scaled_data, sequence_length):

    """
    Aim : we want to change the format of the data to have an array of list of length sequence_length
    and another array of the shifted values which are the price we want to predict 

    Input : 
    scaled_data : type pandas DataFrame
    sequence_length : type int wich is the length of the slice

    Output : Two numpy arrays

    """

   
    xs, ys = [], []
    # Extract the column of data we want to predict (The close price)
    data=scaled_data.iloc[:, 4]
    # Iterate through the data to create sequences
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)] # Input sequence
        y = data[i + sequence_length] # Target value
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def create_sequences_recursive(scaled_data, sequence_length):

    """
    Aim : we want to change the format of the data to have an array of list of length sequence_length
    and another array of the shifted values which are the price we want to predict 

    Input : 
    scaled_data : type pandas DataFrame
    sequence_length : type int wich is the length of the slice

    Output : Two numpy arrays

    """

   
    xs, ys = [], []
    # Extract the column of data we want to predict (The close price)
    data=scaled_data.iloc[:, 4][:-5]
    # Iterate through the data to create sequences
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)] # Input sequence
        y = data[i + sequence_length] # Target value
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def visualize_RNN_prediction(y_train, y_test,predicted_values):
    # Combining y_train and y_test
    full_y = np.concatenate([y_train, y_test])

    # Creating a time axis for the full dataset
    time_steps = np.arange(len(full_y))

    # Determine the starting point for y_test in the combined array
    test_start = len(y_train)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot y_train part
    plt.plot(time_steps[:test_start], full_y[:test_start], label='Real price', color='blue')

    # Plot y_test part
    plt.plot(time_steps[test_start:], full_y[test_start:], label='Real price', color='orange')

    # Plot predicted_values on top of y_test
    plt.plot(time_steps[test_start:], predicted_values, label='Predicted Values', color='green', linestyle='--')

    plt.title('Full Data with Real price and Predicted price')
    plt.xlabel('time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def lstm_model(X, y):

    """

    Input : 
    X, y : type numpy array X is teh realised prices and y the prices we want to learn the model with 

    Output : 
    y_train, y_test : numpy array jsut in order to plot the them to compare properly 
    predicted_values : numpy array the prediction of the model on the test set

    
    """

  
    
    # We split the data into training and testing sets
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)), # LSTM layer with 50 units and return sequences
    Dropout(0.2), # Dropout layer to prevent overfitting
    LSTM(50, return_sequences=False), 
    Dropout(0.2),
    Dense(25), 
    Dense(1) 
    ])

    model.compile(optimizer='adam', loss='mean_squared_error') # Use Adam optimizer and mean squared error loss to optimize the prediction
    model.fit(X_train, y_train, batch_size=351, epochs=100) # Train for 200 epochs (= How many times the entire dataset is used for training) with a batch size (=How many data samples are processed at a time during an epoch) of 351
    predicted_values= model.predict(X_test)

    return y_train, y_test, predicted_values



def recursive_prediction(X,y, t):

 
    """

    Aim : Predict the future prices with LSTM model 
    we predict the next value and we give it to the model for forcasting the next value recursively

    Input : 
    X, y : type numpy array of the prices X the history and y the price we want to learn the model with 
    t : type int number of periods we want to predict 

    Output : type list wich contains the realised prices and the future prices at the end of the list

    """
    
    model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)), # LSTM layer with 50 units and return sequences
    Dropout(0.2), # Dropout layer to prevent overfitting
    LSTM(50, return_sequences=False), 
    Dropout(0.2),
    Dense(25), 
    Dense(1) 
    ])

    model.compile(optimizer='adam', loss='mean_squared_error') # Use Adam optimizer and mean squared error loss to optimize the prediction
    model.fit(X, y, batch_size=351, epochs=100) # Train for 200 epochs (= How many times the entire dataset is used for training) with a batch size (=How many data samples are processed at a time during an epoch) of 351
    
    prediction = y[-X.shape[1]:].tolist()

    while len(prediction) - X.shape[1] < t:
        l = np.array([prediction[-X.shape[1]:]])
        p = model.predict(l)
        prediction.append(p[0][0])
    return prediction


def inverse_scalling(x, data):



    """

    Aim : Inverse scale the data to find the original format of the data after prediction

    Input : 
    x : type numpy array which contains the data to inverse scale 
    data : pandas DataFrame the original data with unscaled values 

    Output : numpy array with the unscaled values

    """

    
    y = pd.DataFrame(x)
    inv_scale  = lambda z : z * (data['high'].max() - data['low'].min()) + data['low'].min()
    x = np.array(y.apply(inv_scale))

    return x


def garch_analysis(data, p, q, dist='normal', window_realized=10, last_n_days=60):
    """
    Analyse GARCH pour un tableau de données donné sans afficher les étapes intermédiaires.
    """
    # Étape 1 : Calcul des rendements logarithmiques
    data = data.copy()  # S'assurer que le DataFrame d'origine reste inchangé
    data.loc[:, 'returns'] = np.log(data['close'] / data['close'].shift(1))
    data = data.dropna()
    
    # Mise à l'échelle des rendements
    scaler = StandardScaler()
    data.loc[:, 'scaled_returns'] = scaler.fit_transform(data['returns'].values.reshape(-1, 1))
    
    # Ajuster le modèle GARCH
    model = arch_model(data['scaled_returns'], vol='Garch', p=p, q=q, dist=dist)
    garch_fit = model.fit(update_freq=5, disp="off")  # Suppression des affichages
    
    # Prédictions pour les derniers jours
    forecast = garch_fit.forecast(horizon=1, start=data.index[-last_n_days])
    data.loc[:, 'predicted_volatility_scaled'] = np.nan
    data.loc[forecast.variance.index, 'predicted_volatility_scaled'] = np.sqrt(forecast.variance.values[:, 0])
    
    # Calculer la volatilité réalisée
    data.loc[:, 'realized_volatility'] = data['scaled_returns'].rolling(window=window_realized).std()
    
    # Visualisation des résultats
    data_nor = data.iloc[-(last_n_days * 2):]
    data_last_n = data.iloc[-last_n_days:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(data_nor.index, data_nor['realized_volatility'], label="Realized Volatility", color='blue')
    plt.plot(data_last_n.index, data_last_n['predicted_volatility_scaled'], label="Predicted Volatility", color='orange')
    plt.title("Realized vs Predicted Volatility (GARCH)")
    plt.xlabel("Date")
    plt.ylabel("Scaled Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calcul des métriques
    realized_volatility = data_last_n['realized_volatility'].dropna()
    predicted_volatility = data_last_n['predicted_volatility_scaled'].dropna()
    
    # Aligner les indices
    realized_volatility = realized_volatility.loc[predicted_volatility.index]
    
    mse = mean_squared_error(realized_volatility, predicted_volatility)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(realized_volatility, predicted_volatility)
    r2 = r2_score(realized_volatility, predicted_volatility)
    
    # Affichage des métriques
    print("Évaluation des prédictions de volatilité :")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

def garch_analysis_for_dashboard(data, p, q, dist='normal', window_realized=10, last_n_days=60):
    """
    Analyse GARCH pour extraire les informations nécessaires au Dashboard.
    Renvoie les données de volatilité réalisée et prédite.
    """
    data = data.copy()
    data.loc[:, 'returns'] = np.log(data['close'] / data['close'].shift(1))
    data = data.dropna()
    
    # Mise à l'échelle des rendements
    scaler = StandardScaler()
    data.loc[:, 'scaled_returns'] = scaler.fit_transform(data['returns'].values.reshape(-1, 1))
    
    # Ajuster le modèle GARCH
    model = arch_model(data['scaled_returns'], vol='Garch', p=p, q=q, dist=dist)
    garch_fit = model.fit(update_freq=5, disp="off")
    
    # Prédictions pour les derniers jours
    forecast = garch_fit.forecast(horizon=1, start=data.index[-last_n_days])
    data.loc[:, 'predicted_volatility_scaled'] = np.nan
    data.loc[forecast.variance.index, 'predicted_volatility_scaled'] = np.sqrt(forecast.variance.values[:, 0])
    
    # Calculer la volatilité réalisée
    data.loc[:, 'realized_volatility'] = data['scaled_returns'].rolling(window=window_realized).std()
    
    # Récupérer les derniers jours de données
    data_last_n = data.iloc[-last_n_days:]
    
    return data_last_n
