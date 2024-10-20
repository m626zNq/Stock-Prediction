# Stock Price Prediction with Machine Learning

This project uses machine learning to predict stock prices based on historical data. It employs a Long Short-Term Memory (LSTM) neural network model to forecast future stock prices for various companies.

## Features

- Data collection from Yahoo Finance using the `yfinance` library
- LSTM model for time series prediction
- Training on multiple stock data to improve generalization
- Prediction visualization with matplotlib
- GPU support for faster training (if available)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- yfinance
- pandas
- numpy
- matplotlib
- scikit-learn

You can install the required packages using:
```
pip install -r requirements.txt
```


## Usage

### Training the Model
<sub>***The weights are already trained and saved in the repo 'weights' folder for the current version. This is optional.***</sub>

To train the model on historical stock data:
```
python scripts/train.py
```

This script will:
1. Download historical stock data for predefined tickers
2. Prepare the data for training
3. Build and train the LSTM model
4. Save the trained model as `model.h5` and the scaler as `scaler.npy`

### Making Predictions

To make predictions using the trained model:
```
python scripts/inference.py --stock <stock symbol> --output <output filename>
```


By default, this will predict stock prices for NVIDIA (NVDA) and save the plot as `predictions.png`. You can modify the ticker and output filename by passing the appropriate arguments as shown above.

## How it Works

1. **Data Collection**: Historical stock data is fetched using the `yfinance` library.
2. **Data Preprocessing**: The data is scaled using MinMaxScaler to normalize the values.
3. **Model Architecture**: An LSTM-based neural network is used for sequence prediction.
4. **Training**: The model is trained on multiple stock data to capture general market trends.
5. **Prediction**: The trained model predicts future stock prices based on recent data.
6. **Visualization**: Predictions are plotted against actual prices, including a simple trading recommendation.

## Customization

- To predict for different stocks, modify the ticker in the `main()` function of `inference.py`.
- To train on different stocks, update the list of tickers in `train.py`.
- Adjust the `time_step` and `future_days` parameters in both scripts to change the input sequence length and prediction horizon.

## License

This project is released under the Unlicense. For more information, see the [LICENSE](LICENSE) file.

## Disclaimer

This may provide inaccurate predictions. Dont rely on it too much until it is improved.