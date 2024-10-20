import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

def get_historical_stock_data(tickers, start_date="2015-01-01", end_date="2023-01-01"):
    all_stock_data = []
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            print(f"No data for {ticker}, skipping.")
            continue
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        all_stock_data.append(stock_data)
    return all_stock_data

def prepare_data_from_multiple_stocks(all_stock_data, time_step=120, future_step=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train, y_train = [], []
    
    all_data = np.concatenate([data.values for data in all_stock_data], axis=0)
    scaler.fit(all_data)
    
    for stock_data in all_stock_data:
        scaled_data = scaler.transform(stock_data)
        
        for i in range(time_step, len(scaled_data) - future_step):
            X_train.append(scaled_data[i-time_step:i])
            y_train.append(scaled_data[i:i+future_step, 3])

    X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train, scaler

def build_model(input_shape, output_shape):
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=100, return_sequences=False),
        Dropout(0.2),
        Dense(units=50, activation='relu'),
        Dense(units=output_shape)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    return model

def train_models(tickers):
    all_stock_data = get_historical_stock_data(tickers)

    if not all_stock_data:
        print("No valid stock data available to train on.")
        return

    X_train, y_train, scaler = prepare_data_from_multiple_stocks(all_stock_data)

    if len(X_train) == 0 or len(y_train) == 0:
        print("Not enough data to train the model.")
        return

    model = build_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    model.fit(X_train, y_train, batch_size=32, epochs=15, validation_split=0.2, verbose=1)

    model.save('weights/model.h5')
    np.save('weights/scaler.npy', scaler)
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_models([
        'AAPL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'MSFT', 'ETH-USD',
        'META', 'NFLX', 'BA', 'JPM', 'V', 'WMT', 'DIS', 'BABA',
        'KO', 'PFE', 'XOM', 'ADBE', 'CSCO', 'ORCL', 'INTC',
        'NFLX', 'CRM', 'IBM', 'NKE', 'QCOM', 'T', 'MCD',
        'CVX', 'LLY', 'TXN', 'AMD', 'HON',  'PYPL', 'SBUX', 'ABNB', 'TWTR', 'SHOP', 'SPCE', 'LMT', 'GILD', 'ZM', 
        'SQ', 'CAT', 'MRK', 'DELL', 'EA', 'SPOT', 'F', 'GM', 'RBLX', 'UBER', 
        'LYFT', 'BMY', 'ABT', 'K', 'TGT', 'SNE', 'ATVI', 'SIVB', 'PINS', 
        'WBA', 'PEP', 'TSM', 'BBY', 'DUK', 'CVS', 'RTX', 'DE', 'VZ', 'MO', 
        'PG', 'UNH', 'CL', 'C', 'LUV', 'AXP', 'HAL', 'GLW', 'MMM', 'ROKU', 
        'SWKS', 'KO', 'MRNA', 'LRCX', 'ZS', 'DHR', 'COST', 'MDT', 'GOOGL', 
        'WFC', 'GE', 'CCL', 'FDX', 'UL', 'VEEV', 'ANTM', 'TTD', 'BB', 'PLTR', 
        'KHC', 'GME', 'BBVA', 'UBS', 'PDD', 'NOK', 'DOCU', 'SAP', 'KMI', 
        'BAX', 'BIDU', 'TMUS', 'AVGO', 'RCL', 'HOG', 'AAL', 'ENB', 'NOC', 
        'KO', 'SNPS', 'NOW', 'DOW', 'SCHW', 'EOG', 'MSCI', 'TTWO', 'STZ', 
        'NVAX', 'GPN', 'FIS', 'MPC', 'DG', 'LULU', 'SOL-ETH'
    ])