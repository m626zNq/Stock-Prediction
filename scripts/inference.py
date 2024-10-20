import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(e)

def get_stock_data(ticker):
    stock_data = yf.download(ticker, period="6mo", interval="1d")
    return stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]

def predict_stock_price(model, data, scaler, time_step=120, future_days=30):
    data_scaled = scaler.transform(data)
    last_time_step_data = data_scaled[-time_step:]
    future_predictions = []

    input_data = np.reshape(last_time_step_data, (1, time_step, data.shape[1]))
    predicted_prices = model.predict(input_data)[0]
    
    last_known_price = data[-1, 3]
    future_predictions.append(last_known_price)
    
    for price in predicted_prices:
        scaled_price = np.array([[0, 0, 0, price, 0]])
        unscaled_price = scaler.inverse_transform(scaled_price)[0, 3]
        future_predictions.append(unscaled_price)

    return np.array(future_predictions)

def plot_and_save_predictions(actual, future_predictions, ticker, filename="predictions.png"):
    plt.figure(figsize=(12, 6))
    future_index = np.arange(len(actual) - 1, len(actual) - 1 + len(future_predictions))
    plt.plot(np.arange(len(actual)), actual[:, 3], label="Actual Price", color="green", linewidth=2)
    plt.plot(future_index, future_predictions, label="Future Predictions", color="blue", linestyle="--", linewidth=2)
    plt.axvline(x=len(actual) - 1, color="gray", linestyle="--")
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time (days)')
    plt.ylabel('Price')
    plt.legend()

    last_actual_price = actual[-1, 3]
    final_predicted_price = future_predictions[-1]

    if final_predicted_price > last_actual_price * 1.05:
        recommendation = "BUY NOW"
    elif final_predicted_price < last_actual_price * 0.95:
        recommendation = "SELL NOW"
    else:
        recommendation = "HOLD"

    plt.figtext(0.5, 0.01, recommendation, ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def main(ticker='AAPL', output_image='predictions.png'):
    custom_objects = {"MeanSquaredError": MeanSquaredError}
    
    model = load_model('weights/model.h5', custom_objects=custom_objects)
    scaler = np.load('weights/scaler.npy', allow_pickle=True).item()
    stock_data = get_stock_data(ticker)

    future_predictions = predict_stock_price(model, stock_data.values, scaler)

    plot_and_save_predictions(stock_data.values, future_predictions, ticker, filename=output_image)
    print(f"Predictions saved to {output_image}")

if __name__ == "__main__":
    main('NVDA', 'predictions.png')