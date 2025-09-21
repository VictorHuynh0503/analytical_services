import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import date

# ---------------------------------------------------
# 1. Standardize dataframe
# ---------------------------------------------------
def standardize_df(df, df_info, df_nganh, start_date=date(2024,1,1)):
    df["Date"] = pd.to_datetime(df["Datetime"]).dt.date
    df = df[df["Date"] >= start_date]

    df = df.rename(columns={
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume"
    })
    df = df['Ticker,Date,Open,High,Low,Close,Volume'.split(',')]
    
    # merge company info
    df = df.merge(df_info[['stock_code', 'name_vn']], 
                  left_on='Ticker', right_on='stock_code', how='left')
    df = df.merge(df_nganh[['ticker', 'industry']], 
                  left_on='Ticker', right_on='ticker', how='left')
    
    return df

# ---------------------------------------------------
# 2. Add indicators
# ---------------------------------------------------
def add_indicators(df):
    df["rsi"] = ta.rsi(df["Close"], length=14)
    df["ema20"] = ta.ema(df["Close"], length=20)
    df["ema50"] = ta.ema(df["Close"], length=50)
    df = df.dropna()
    return df

# ---------------------------------------------------
# 3. Plot candlestick chart
# ---------------------------------------------------
def plot_candlestick(df, ticker="Stock"):
    mpf.plot(
        df.set_index("Date"),  # ensure index is datetime
        type="candle",
        style="charles",
        title=f"{ticker} OHLC with EMA20 & EMA50",
        mav=(20, 50),
        volume=True,
        figsize=(12, 8)
    )

# ---------------------------------------------------
# 4. Prepare data for LSTM
# ---------------------------------------------------
def prepare_lstm_data(df, feature="Close", seq_len=60):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[[feature]].values)

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i-seq_len:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# ---------------------------------------------------
# 5. Build LSTM model
# ---------------------------------------------------
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # output layer
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ---------------------------------------------------
# 6. Train + Predict + Plot
# ---------------------------------------------------
def train_and_predict(df, feature="Close", seq_len=60, epochs=20, batch_size=32):
    X, y, scaler = prepare_lstm_data(df, feature, seq_len)

    model = build_lstm((X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    predicted = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted)
    real_prices = scaler.inverse_transform(y.reshape(-1,1))

    plt.figure(figsize=(12,6))
    plt.plot(real_prices, color="black", label="Real Price")
    plt.plot(predicted_prices, color="green", label="Predicted Price")
    plt.title("LSTM Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    return model, predicted_prices, real_prices

# ---------------------------------------------------
# Usage Example
# ---------------------------------------------------
# df_final = standardize_df(df, df_info, df_nganh)
# df_final = add_indicators(df_final)
# plot_candlestick(df_final, ticker="ABC")
# model, preds, real = train_and_predict(df_final, feature="Close")
