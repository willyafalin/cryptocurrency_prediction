#Libraries and modules 

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



# Function to extract historical data from Binance
def fetch_binance_data(pair, freq, start, end):
    """
    Purpose: Retrieve historical cryptocurrency data from Binance within a specified time range.
    
    Input:
    - pair: type string, the trading pair (e.g., 'BTC/USD')
    - freq: type string, frequency of data (e.g., '1day', '5min')
    - start: type string, start date in 'YYYY-MM-DD' format
    - end: type string, end date in 'YYYY-MM-DD' format
    
    Output: 
    - data_frame: pandas DataFrame containing historical data including 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    """
    
    binance_client = Client()  # Set up Binance client
    
    # Convert dates to timestamps
    start_timestamp = int(pd.Timestamp(start).timestamp() * 1000)
    end_timestamp = int(pd.Timestamp(end).timestamp() * 1000)
    historical_data = []

    # Download data in batches (maximum 1000 candles per request)
    while start_timestamp < end_timestamp:
        data_batch = binance_client.get_klines(
            symbol=pair,
            interval=freq,
            limit=1000,
            startTime=start_timestamp
        )
        if not data_batch:
            break
        historical_data.extend(data_batch)
        start_timestamp = data_batch[-1][0] + 1  # Move to the next batch
        time.sleep(0.1)  # Adhere to API rate limits

    # Convert data into a DataFrame
    data_frame = pd.DataFrame(historical_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    data_frame['timestamp'] = pd.to_datetime(data_frame['timestamp'], unit='ms')  # Convert to a readable format
    data_frame[['open', 'high', 'low', 'close', 'volume']] = data_frame[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return data_frame


# Function to visualize data
def plot_cryptocurrency(data_frame, currency_pair, time_interval):
    """
    Purpose: Plot the closing price of the selected cryptocurrency over the specified interval
    
    Input: 
    - data_frame: type pandas DataFrame
    - currency_pair: type string, for instance 'BTC/USD'
    - time_interval: type string, for instance '1day' or '5min'
    
    Output: Plot
    """
    
    sns.set(style='darkgrid')
    plt.title('Closing Price of ' + str(currency_pair))
    plt.xlabel(str(time_interval))
    plt.ylabel('USD')
    # Visualize the closing price
    sns.lineplot(x=data_frame.index, y=data_frame['close'], color='green')
    plt.show()


# Function to visualize financial data using candlestick chart
def render_market_candles(data_frame, currency, interval):
    """
    Purpose: Display the candlestick chart for the chosen cryptocurrency over the specified interval
    
    Input: 
    - data_frame: type pandas DataFrame
    - currency: type string, for instance 'BTC/USD'
    - interval: type string, for instance '1day' or '5min'
        
    Output: Plot
    """

    # Create a candlestick chart using Plotly
    fig = go.Figure(data=[go.Candlestick(x=data_frame['timestamp'],
                                     open=data_frame['open'],
                                     high=data_frame['high'],
                                     low=data_frame['low'],
                                     close=data_frame['close'])])
    
    fig.update_layout(title='Market Chart of ' + str(currency) + ' per ' + str(interval),
                  xaxis_title='Date',
                  yaxis_title='Price in USD',
                  xaxis_rangeslider_visible=False)

    fig.show()


# Function to visualize market data with an indicator
def plot_with_indicator(data_frame, currency_pair, time_frame, tech_indicator):
    """
    Purpose: Display the candlestick chart for the selected cryptocurrency
    over the specified interval, including a technical indicator
    
    Input: 
    - data_frame: type pandas DataFrame
    - currency_pair: type string, for instance 'BTC/USD'
    - time_frame: type string, for instance '1day' or '5min'
    - tech_indicator: type string, RSI, EMA, ATR
        
    Output: Plot
    """
 
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Candles', str(tech_indicator)])
    
    # Create candlestick chart
    candle_chart = go.Candlestick(x=data_frame['timestamp'],
                                  open=data_frame['open'],
                                  high=data_frame['high'],
                                  low=data_frame['low'],
                                  close=data_frame['close'])

    fig.add_trace(candle_chart, row=1, col=1)

    # Add technical indicator trace
    indicator_trace = go.Scatter(x=data_frame['timestamp'], y=data_frame[str(tech_indicator)], mode='lines', name=str(tech_indicator), yaxis='y2')

    fig.add_trace(indicator_trace, row=2, col=1)

    fig.update_layout(title='Market Chart of ' + str(currency_pair) + ' per ' + str(time_frame),
                      xaxis_title='Date',
                      yaxis_title='Price in USD',
                      xaxis_rangeslider_visible=False)

    fig.show()




# Function to add technical indicators to the data
def append_technical_indicators(df, window=14):

    """
    Purpose: Compute and integrate three technical indicators (RSI, EMA, and ATR) into a pandas DataFrame.
    
    Input:
    - df: type pandas DataFrame 

    Output: 
    - df: pandas DataFrame enriched with the three indicators (RSI, EMA, ATR)
    """
    
    # Compute Exponential Moving Average (EMA)
    ema = ta.trend.ema_indicator(close=df['close'], window=window).dropna()
    
    # Compute Relative Strength Index (RSI)
    rsi = ta.momentum.rsi(close=df['close'], window=window).dropna()
    
    # Compute Average True Range (ATR)
    atr = ta.volatility.AverageTrueRange(close=df['close'], high=df['high'], low=df['low'], window=window).average_true_range()
    atr = atr[atr > 0]
    
    # Adjust the DataFrame to align with the window size
    df = pd.DataFrame(df.loc[window-1:])
    
    # Incorporate the computed indicators into the DataFrame
    df['RSI'] = rsi
    df['EMA'] = ema
    df['ATR'] = atr

    return df.reset_index().drop('index', axis=1)


# Function to scale the data using MinMax Scaler
def apply_minmax_scaler(df):

    """
    Purpose: Normalize the DataFrame using MinMax Scaler for machine learning models
    
    Input: 
    - df: type pandas DataFrame

    Output: 
    - scaled_df: pandas DataFrame with normalized values
    """
    
    # Duplicate the data to avoid altering the original DataFrame
    scaled_df = df.copy()

    # Identify the global minimum and maximum values from the 'low' and 'high' columns, respectively
    global_min = df['low'].min()
    global_max = df['high'].max()

    # Define a function to normalize each value
    def scale_value(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    # Normalize the 'open' and 'close' prices relative to the global min and max
    scaled_df['open'] = df['open'].apply(scale_value, args=(global_min, global_max))
    scaled_df['close'] = df['close'].apply(scale_value, args=(global_min, global_max))

    # Directly normalize 'high' and 'low' with global min and max as well
    scaled_df['high'] = df['high'].apply(scale_value, args=(global_min, global_max))
    scaled_df['low'] = df['low'].apply(scale_value, args=(global_min, global_max))

    # Normalize other columns individually 
    for column in df.columns:
        if column not in ['timestamp', 'open', 'high', 'low', 'close']:
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            # Normalize each column separately
            scaled_column = scaler.fit_transform(df[[column]])
            # Store the normalized values back in the DataFrame
            scaled_df[column] = scaled_column.flatten()

    return scaled_df


# Function to preprocess data for time series regression
def preprocess_for_regression(scaled_df, feature_columns, future_steps):

    """  
    Purpose: Adjust the data to conduct regression on time series and generate feature-target pairs.
    
    Input:
    - scaled_df: type pandas DataFrame, normalized data 
    - feature_columns: type list, columns to be used as features
    - future_steps: type int, number of steps to shift for predicting future values

    Output:
    - prices: type numpy array, features for regression 
    - targets: type numpy array, targets for regression
    """
   
    # Shift the 'close' prices to form targets
    targets = scaled_df['close'].shift(-future_steps).dropna()
    targets = np.array(targets).reshape(-1, 1)

    # Create features array by excluding the last 'future_steps' rows
    prices = np.array(scaled_df[feature_columns])[:-future_steps]
    
    return prices, targets


# Function to visualize model predictions
def plot_model_predictions(predictions_df, scaled_df, zoom_range=None):
    """
    Purpose: Visualize the actual versus predicted prices for the given data

    Input: 
    - predictions_df: type pandas DataFrame, DataFrame containing the predictions
    - scaled_df: type pandas DataFrame, normalized data
    - zoom_range: type tuple, range to zoom in the plot (optional)

    Output: None (Generates a plot)
    """

    plt.xlabel('Days')
    plt.ylabel('BTC/USD ($)(scaled data)')
    
    # Visualize real close prices
    plt.plot(scaled_df['close'])
    
    # Visualize predicted close prices
    plt.plot(predictions_df[['close', 'prediction']])
    
    plt.legend(['Real Price', 'Prediction'])
    
    if zoom_range is not provided: 
        plt.xlim(zoom_range[0], zoom_range[1])
    
    plt.title('Prediction of Close Price of BTC/USD for the Last Month by Linear Regression')
    plt.show()



# Function to apply linear regression on the data
def apply_linreg(scaled_df, future_steps, prices, targets, features):
    """
    Purpose: Utilize linear regression to forecast future prices and assess the model.
    
    Input: 
    - scaled_df: type pandas DataFrame, the normalized data 
    - future_steps: type int, the number of future steps to predict 
    - prices: type numpy array, the features for training the model
    - targets: type numpy array, the targets for training the model
    - features: type list of strings, list of feature columns 

    Output: 
    - prediction_df: type pandas DataFrame, containing the realized prices and predictions 
    - future_preds: type numpy array, containing the future prices from the last date in the dataset 
    - r2: type float, R2 score of the linear regression model
    """

    # Divide the data into training and testing sets
    prices_train, prices_test, targets_train, targets_test = train_test_split(prices, targets, test_size=0.3)
    
    # Train the linear regression model on the training set 
    linreg_model = LinearRegression().fit(prices_train, targets_train)

    # Forecast the last segment of prices
    prices_to_predict = prices[-future_steps:]
    linreg_predictions = linreg_model.predict(prices_to_predict)

    # Construct a DataFrame to hold the realized prices 
    prediction_df = pd.DataFrame(scaled_df['close'].tail(future_steps))
    # Include the predictions as a new column
    prediction_df['prediction'] = linreg_predictions

    # Forecast future values from the last segment of data
    future_prices = np.array(scaled_df[features])[-future_steps:]
    future_preds = linreg_model.predict(future_prices)

    # Forecast targets on the test set and compute the R2 score
    targets_predicted = linreg_model.predict(prices_test)
    r2 = r2_score(targets_test, targets_predicted)

    return prediction_df, future_preds, r2


# Function to visualize future price predictions
def plot_future_predictions(scaled_df, future_preds, zoom_range=None):
    """
    Purpose: Display the future price predictions alongside the historical prices.
    
    Input: 
    - scaled_df: type pandas DataFrame, containing historical normalized data
    - future_preds: type numpy array, containing the predicted future prices
    - zoom_range: type tuple, optional, range to zoom in the plot

    Output: None (Generates a plot)
    """

    plt.xlabel('Days')
    plt.ylabel('BTC/USD ($)(scaled data)')
    
    # Convert the 'close' prices and future predictions to numpy arrays
    close_prices = np.array(scaled_df['close']).reshape(-1, 1)
    future_prices = np.array(future_preds).reshape(-1, 1) 
    combined_prices = np.concatenate((close_prices, future_prices))
    
    # Draw a vertical line indicating the start of predictions
    plt.axvline(x=close_prices.shape[0], color='r', linestyle='--', label='Prediction')
    plt.plot(combined_prices)
    
    if zoom_range is provided: 
        plt.xlim(zoom_range[0], zoom_range[1])
    
    plt.title('Prediction of Close Price of BTC/USD')
    plt.show()


# Function to apply Support Vector Regression on the data
def apply_svr_model(scaled_df, future_steps, prices, targets, feature_list, optimal_C, optimal_gamma):
    """
    Purpose: Use SVR to predict future prices based on historical data and assess model performance.
    
    Input: 
    - scaled_df: type pandas DataFrame, containing the normalized data 
    - future_steps: type int, the number of future values to predict 
    - prices: type numpy array, the features for training the model
    - targets: type numpy array, the targets for training the model
    - feature_list: type list of strings, columns to be used as features 
    - optimal_C: type float, optimal C parameter for the SVR model
    - optimal_gamma: type float, optimal gamma parameter for the SVR model 

    Output: 
    - prediction_df: type pandas DataFrame, containing the realized prices and predictions 
    - future_preds: type numpy array, containing the predicted future prices from the last date of the dataset 
    - svr_accuracy: type float, measure of the model's accuracy (R² score)
    """

    # Divide the data into training and testing sets
    prices_train, prices_test, targets_train, targets_test = train_test_split(prices, targets, test_size=0.3)
    
    # Set up and train the SVR model
    svr_rbf = SVR(kernel='rbf', C=optimal_C, gamma=optimal_gamma)
    svr_rbf.fit(prices_train, np.ravel(targets_train))

    # Forecast the last segment of prices
    prices_to_predict = prices[-future_steps:] 
    svr_predictions = svr_rbf.predict(prices_to_predict)

    # Construct a DataFrame to hold the realized prices 
    prediction_df = pd.DataFrame(scaled_df['close'].tail(future_steps))
    prediction_df['prediction'] = svr_predictions

    # Forecast future values from the last segment of data
    future_prices = np.array(scaled_df[feature_list])[-future_steps:]
    future_preds = svr_rbf.predict(future_prices)

    # Evaluate the SVR model accuracy (R² score)
    svr_accuracy = svr_rbf.score(prices_test, targets_test)

    return prediction_df, future_preds, svr_accuracy


# Function to find the best parameters C and Gamma by Cross Validation
def find_best_hyperparameters(param_grid, prices_train, targets_train):
    """
    Purpose: Identify the optimal parameters C and Gamma through Cross Validation.
    
    Input: 
    - param_grid: type dictionary, containing the values of gamma and C to evaluate 
    - prices_train: type numpy array, training features
    - targets_train: type numpy array, training targets

    Output: 
    - best_C: type float, optimal C parameter
    - best_gamma: type float, optimal gamma parameter
    """

    # Use Support Vector Regression
    svr_rbf = SVR(kernel='rbf')
    # Conduct grid search with cross-validation
    search = GridSearchCV(svr_rbf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    search.fit(prices_train, np.ravel(targets_train))  # Fit cross-validation on the data
    
    # Retrieve the optimal parameters
    best_C = search.best_params_['C']
    best_gamma = search.best_params_['gamma']

    return best_C, best_gamma



# Function to create sequences for training the model
def generate_sequences(df, seq_length):
    """
    Purpose: Reformat the data to generate sequences of a specified length 
    and their corresponding target values for model training.

    Input: 
    - df: type pandas DataFrame, the input data
    - seq_length: type int, the length of each sequence

    Output: 
    - sequences: type numpy array, input sequences of length seq_length
    - targets: type numpy array, corresponding target values
    """

    xs, ys = [], []
    # Select the column of data to be predicted (The close price)
    target_data = df.iloc[:, 4]
    # Loop through the data to create sequences
    for i in range(len(target_data) - seq_length):
        x = target_data[i:(i + seq_length)]  # Input sequence
        y = target_data[i + seq_length]  # Target value
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Function to visualize RNN predictions
def plot_rnn_predictions(train_values, test_values, predictions):
    """
    Purpose: Display the real and predicted prices using RNN model predictions.

    Input: 
    - train_values: numpy array, the real training prices
    - test_values: numpy array, the real testing prices
    - predictions: numpy array, the predicted prices by the model

    Output: None (Generates a plot)
    """
    
    # Combine train and test values
    combined_y = np.concatenate([train_values, test_values])

    # Create a time axis for the full dataset
    time_axis = np.arange(len(combined_y))

    # Identify the starting point for test_values in the combined array
    test_start_idx = len(train_values)

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot train values
    plt.plot(time_axis[:test_start_idx], combined_y[:test_start_idx], label='Real Price (Train)', color='blue')

    # Plot test values
    plt.plot(time_axis[test_start_idx:], combined_y[test_start_idx:], label='Real Price (Test)', color='orange')

    # Plot predicted values over test values
    plt.plot(time_axis[test_start_idx:], predictions, label='Predicted Values', color='green', linestyle='--')

    plt.title('Full Data with Real and Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Function to create and train an LSTM model for time series prediction
def build_lstm_model(X, y):
    """
    Purpose: Develop and train an LSTM model on the provided data to forecast time series values.

    Input: 
    - X, y: type numpy array, X represents the input sequences and y represents the target values

    Output: 
    - y_train, y_test: numpy arrays, utilized to plot and compare predictions 
    - predicted_values: numpy array, the model's predictions on the test set
    """
    
    # Partition the data into training and testing sets
    train_size = int(len(X) * 0.7)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Construct the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),  # LSTM layer with 50 units and return sequences
        Dropout(0.2),  # Dropout layer to mitigate overfitting
        LSTM(50, return_sequences=False),  # Second LSTM layer
        Dropout(0.2),  # Second Dropout layer
        Dense(25),  # Dense layer with 25 units
        Dense(1)  # Output layer with a single unit
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')  # Use Adam optimizer and mean squared error loss
    model.fit(X_train, y_train, batch_size=351, epochs=100)  # Train the model for 100 epochs with a batch size of 351
    predicted_values = model.predict(X_test)  # Predict on the test set

    return y_train, y_test, predicted_values

# Function to predict future prices with LSTM model using recursive prediction
def recursive_lstm_prediction(X, y, future_steps):
    """
    Purpose: Forecast future prices using LSTM model. 
    The model predicts the next value and utilizes it to forecast subsequent values recursively.

    Input: 
    - X, y: type numpy array, X represents the historical prices and y represents the prices we want to train the model with 
    - future_steps: type int, number of periods we want to predict 

    Output: type list containing the realized prices and the future prices at the end of the list
    """
    
    # Construct the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),  # LSTM layer with 50 units and return sequences
        Dropout(0.2),  # Dropout layer to mitigate overfitting
        LSTM(50, return_sequences=False),  # Second LSTM layer
        Dropout(0.2),  # Second Dropout layer
        Dense(25),  # Dense layer with 25 units
        Dense(1)  # Output layer with a single unit
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')  # Use Adam optimizer and mean squared error loss
    model.fit(X, y, batch_size=351, epochs=100)  # Train for 100 epochs with a batch size of 351
    
    # Initialize predictions with the last segment of y
    prediction = y[-X.shape[1]:].tolist()

    # Recursively forecast future prices
    while len(prediction) - X.shape[1] < future_steps:
        input_sequence = np.array([prediction[-X.shape[1]:]])
        next_prediction = model.predict(input_sequence)
        prediction.append(next_prediction[0][0])
    
    return prediction

# Function to inverse scale the data to find the original format after prediction
def reverse_scaling(predicted_values, original_df):
    """
    Purpose: Reverse scale the data to recover the original format after prediction.

    Input: 
    - predicted_values: type numpy array containing the data to inverse scale 
    - original_df: type pandas DataFrame, the original data with unscaled values 

    Output: numpy array with the unscaled values
    """

    # Convert predicted values to DataFrame
    temp_df = pd.DataFrame(predicted_values)
    
    # Define the reverse scaling function
    inv_scale = lambda z: z * (original_df['high'].max() - original_df['low'].min()) + original_df['low'].min()
    
    # Apply the reverse scaling function to the DataFrame
    unscaled_values = np.array(temp_df.apply(inv_scale))

    return unscaled_values





# Function for GARCH analysis on a given DataFrame
def perform_garch_analysis(df, p, q, distribution='normal', window=10, last_days=60):
    """
    Perform GARCH analysis on a given DataFrame without displaying intermediate steps.

    Input:
    - df: pandas DataFrame
    - p, q: Integers, order of the GARCH model
    - distribution: String, distribution assumption for the model (default: 'normal')
    - window: Integer, window size for realized volatility calculation (default: 10)
    - last_days: Integer, number of last days to include in the forecast (default: 60)

    Output:
    - Visualizes the realized and predicted volatility for the last days
    - Prints evaluation metrics (MSE, RMSE, MAE, R² Score)
    """
    # Step 1: Calculate log returns
    df = df.copy()  # Ensure the original DataFrame remains unchanged
    df.loc[:, 'returns'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()
    
    # Scale the returns
    scaler = StandardScaler()
    df.loc[:, 'scaled_returns'] = scaler.fit_transform(df['returns'].values.reshape(-1, 1))
    
    # Fit the GARCH model
    model = arch_model(df['scaled_returns'], vol='Garch', p=p, q=q, dist=distribution)
    garch_fit = model.fit(update_freq=5, disp="off")  # Suppress output
    
    # Forecast for the last days
    forecast = garch_fit.forecast(horizon=1, start=df.index[-last_days])
    df.loc[:, 'predicted_volatility_scaled'] = np.nan
    df.loc[forecast.variance.index, 'predicted_volatility_scaled'] = np.sqrt(forecast.variance.values[:, 0])
    
    # Calculate realized volatility
    df.loc[:, 'realized_volatility'] = df['scaled_returns'].rolling(window=window).std()
    
    # Visualization
    data_nor = df.iloc[-(last_days * 2):]
    data_last_n = df.iloc[-last_days:]
    
    plt.figure(figsize=(12, 6))
    plt.plot(data_nor.index, data_nor['realized_volatility'], label="Realized Volatility", color='blue')
    plt.plot(data_last_n.index, data_last_n['predicted_volatility_scaled'], label="Predicted Volatility", color='orange')
    plt.title(f"Realized vs Predicted Volatility (GARCH) - Last {last_days} Days")
    plt.xlabel("Date")
    plt.ylabel("Volatility (Scaled)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Calculate metrics
    realized_volatility = data_last_n['realized_volatility'].dropna()
    predicted_volatility = data_last_n['predicted_volatility_scaled'].dropna()
    
    # Align indices
    realized_volatility = realized_volatility.loc[predicted_volatility.index]
    
    mse = mean_squared_error(realized_volatility, predicted_volatility)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(realized_volatility, predicted_volatility)
    r2 = r2_score(realized_volatility, predicted_volatility)
    
    # Print metrics
    print("Volatility Prediction Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")


# Function for GARCH analysis to extract necessary information for the dashboard
def garch_analysis_dashboard(df, p, q, distribution='normal', window=10, last_days=60):

    """
    Perform GARCH analysis to extract necessary information for the dashboard.
    Returns the realized and predicted volatility data.

    Input:
    - df: pandas DataFrame
    - p, q: Integers, order of the GARCH model
    - distribution: String, distribution assumption for the model (default: 'normal')
    - window: Integer, window size for realized volatility calculation (default: 10)
    - last_days: Integer, number of last days to include in the forecast (default: 60)

    Output:
    - pandas DataFrame with realized and predicted volatility for the last days
    """

    df = df.copy()
    df.loc[:, 'returns'] = np.log(df['close'] / df['close'].shift(1))
    df = df.dropna()
    
    # Scale the returns
    scaler = StandardScaler()
    df.loc[:, 'scaled_returns'] = scaler.fit_transform(df['returns'].values.reshape(-1, 1))
    
    # Fit the GARCH model
    model = arch_model(df['scaled_returns'], vol='Garch', p=p, q=q, dist=distribution)
    garch_fit = model.fit(update_freq=5, disp="off")
    
    # Forecast for the last days
    forecast = garch_fit.forecast(horizon=1, start=df.index[-last_days])
    df.loc[:, 'predicted_volatility_scaled'] = np.nan
    df.loc[forecast.variance.index, 'predicted_volatility_scaled'] = np.sqrt(forecast.variance.values[:, 0])
    
    # Calculate realized volatility
    df.loc[:, 'realized_volatility'] = df['scaled_returns'].rolling(window=window).std()
    
    # Retrieve the last days of data
    data_last_n = df.iloc[-last_days:]
    
    return data_last_n





