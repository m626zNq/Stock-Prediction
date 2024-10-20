import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tqdm
import os

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)

def get_historical_stock_data(tickers):
    all_stock_data = []
    for ticker in tqdm.tqdm(tickers, desc="Downloading stock data", unit="ticker", ncols=100, colour="MAGENTA", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} stocks [{elapsed}<{remaining}]"):
        stock_data = yf.download(ticker, period="max", progress=False)
        if stock_data.empty:
            print(f"No data for {ticker}, skipping.")
            continue
        stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

        mid_date = stock_data.index[len(stock_data)//2]
        if (stock_data.index[-1] - stock_data.index[0]).days > 3650: # split the data if its more than 10 years
            all_stock_data.extend([stock_data.loc[:mid_date], stock_data.loc[mid_date:]])
        else:
            all_stock_data.append(stock_data)

    return all_stock_data

def prepare_data_from_multiple_stocks(all_stock_data, time_step=120, future_step=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_data = np.concatenate([data.values for data in all_stock_data], axis=0)
    scaler.fit(all_data)

    X_train, y_train = [], []
    for stock_data in all_stock_data:
        scaled_data = scaler.transform(stock_data.values)
        for i in range(time_step, len(scaled_data) - future_step):
            X_train.append(scaled_data[i-time_step:i])
            y_train.append(scaled_data[i:i+future_step, 3])

    return np.array(X_train), np.array(y_train), scaler

def build_model(input_shape, output_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(128),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(output_shape)
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


    # Create 'weights' directory if it doesn't exist
    if not os.path.exists('weights'):
        os.makedirs('weights')
        print("Created 'weights' directory.")

    # build and save the scaler
    model = build_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    np.save('weights/scaler.npy', scaler)
    print("Scaler saved successfully.")

    # save the model every epoch and use keras instead of .h5 (outdated)
    checkpoint = ModelCheckpoint('weights/model_epoch_{epoch:02d}.keras', save_weights_only=False, save_freq='epoch', save_best_only=False, verbose=1)
    model.fit(X_train, y_train, batch_size=64, epochs=50, validation_split=0.2, verbose=1, callbacks=[checkpoint])
    model.save('weights/model.keras')
    print("Model saved successfully.")

if __name__ == "__main__":
    setup_gpu()
    train_models([
        'AAPL', 'GOOG', 'AMZN', 'TSLA', 'NVDA', 'MSFT', 'ETH-USD',
        'META', 'NFLX', 'BA', 'JPM', 'V', 'WMT', 'DIS', 'BABA',
        'KO', 'PFE', 'XOM', 'ADBE', 'CSCO', 'ORCL', 'INTC',
        'NFLX', 'CRM', 'IBM', 'NKE', 'QCOM', 'T', 'MCD',
        'CVX', 'LLY', 'TXN', 'AMD', 'HON',  'PYPL', 'SBUX', 'ABNB', 'SHOP', 'SPCE', 'LMT', 'GILD', 'ZM',
        'SQ', 'CAT', 'MRK', 'DELL', 'EA', 'SPOT', 'F', 'GM', 'RBLX', 'UBER',
        'LYFT', 'BMY', 'ABT', 'K', 'TGT', 'PINS',
        'WBA', 'PEP', 'TSM', 'BBY', 'DUK', 'CVS', 'RTX', 'DE', 'VZ', 'MO',
        'PG', 'UNH', 'CL', 'C', 'LUV', 'AXP', 'HAL', 'GLW', 'MMM', 'ROKU',
        'SWKS', 'KO', 'MRNA', 'LRCX', 'ZS', 'DHR', 'COST', 'MDT', 'GOOGL',
        'WFC', 'GE', 'CCL', 'FDX', 'UL', 'VEEV', 'TTD', 'BB', 'PLTR',
        'KHC', 'GME', 'BBVA', 'UBS', 'PDD', 'NOK', 'DOCU', 'SAP', 'KMI',
        'BAX', 'BIDU', 'TMUS', 'AVGO', 'RCL', 'HOG', 'AAL', 'ENB', 'NOC',
        'KO', 'SNPS', 'NOW', 'DOW', 'SCHW', 'EOG', 'MSCI', 'TTWO', 'STZ',
        'NVAX', 'GPN', 'FIS', 'MPC', 'DG', 'LULU', 'SOL-ETH',
        'ORCL', 'CRM', 'ADSK', 'INTU', 'WDAY', 'NOW', 'TEAM',
        'DDOG', 'SNOW', 'CRWD', 'ZS', 'OKTA', 'NET', 'FTNT', 'PANW', 'CYBR',
        'RPD', 'TENB', 'VRNS', 'QLYS', 'CRWD',
        'FSLY', 'DOCN', 'APPS', 'PD', 'ESTC', 'TWLO',
        'TTD', 'ROKU', 'MGNI', 'PUBM', 'CRTO', 'APPS', 'PERI', 'BZUN', 'SE',
        'MELI', 'STNE', 'LSPD', 'SHOP', 'BIGC', 'WIX', 'SQSP', 'ETSY', 'FVRR',
        'UPWK', 'EXPE', 'BKNG', 'TRIP', 'ABNB', 'SABR', 'TCOM',
        'BTC-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'DOT1-USD', 'UNI3-USD',
        'BCH-USD', 'LTC-USD', 'LINK-USD', 'XLM-USD', 'MATIC-USD', 'ALGO-USD',
        'XTZ-USD', 'EOS-USD', 'AAVE-USD', 'ATOM1-USD', 'MKR-USD', 'COMP-USD',
        'YFI-USD', 'SNX-USD', 'GRT-USD', 'UMA-USD', 'SUSHI-USD', 'ZRX-USD',
        'BAT-USD', 'REN-USD', 'KNC-USD', 'BNT-USD', 'CRV-USD', 'BAND-USD',
        'NMR-USD', 'ANT-USD', 'REP-USD', 'BAL-USD', 'COMP-USD', '1INCH-USD',
        'AVAX-USD', 'SOL1-USD', 'FTT-USD', 'THETA-USD', 'VET-USD', 'FIL-USD',
        'TRX-USD', 'NEAR-USD', 'HBAR-USD', 'ONE1-USD', 'CHZ-USD', 'HOT1-USD',
        'WAVES-USD', 'KAVA-USD', 'OMG-USD', 'ICX-USD', 'ONT-USD', 'ZIL-USD',
        'AMAT', 'ASML', 'CDNS', 'CHTR', 'CTAS', 'CPRT', 'DXCM', 'ENPH',
        'FAST', 'IDXX', 'ILMN', 'KLAC', 'MCHP', 'MDLZ', 'MNST', 'ODFL',
        'ORLY', 'PCAR', 'REGN', 'ROST', 'SIRI', 'VRSN', 'VRTX',
        'WDAY', 'XEL', 'ZM', 'ZS'])
