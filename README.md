# Stock Price Prediction

## Install
Install packages by running `pip install -r requirements.txt`

## Train
Train the model by running `python train.py`

## Inference
Make predictions by running `python inference.py` (change the stock name at the bottom of the file)

## How it works
The model is trained on historical stock data to predict the closing price of a stock. It uses an LSTM neural network to make predictions.

The training data includes features such as open, high, low, and volume. The target is the closing price.

This model is only trained on historical data.. so dont believe the predictions too much.




