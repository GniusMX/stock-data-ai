import yfinance as yf
import pandas as pd
import numpy as np


# ==============================
# RSI
# ==============================
def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


# ==============================
# MACD
# ==============================
def calculate_macd(close):
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = macd - signal

    return macd, signal, histogram


# ==============================
# ADX / ATR
# ==============================
def calculate_adx_atr(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    # Directional Movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    plus_dm = pd.Series(
        plus_dm,
        index=df.index
    )

    minus_dm = pd.Series(
        minus_dm,
        index=df.index
    )

    # Wilder smoothing
    atr = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    plus_di = (
        100 *
        plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    minus_di = (
        100 *
        minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    dx = (
        100 *
        (plus_di - minus_di).abs()
        / (plus_di + minus_di)
    )

    adx = dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    return adx, atr


# ==============================
# 株価データ取得
# ==============================
stock_tickers = {
    "SMH": "SMH",
    "QQQ": "QQQ"
}

volatility_tickers = {
    "VIX": "^VIX",
    "VIX3M": "^VIX3M"
}


# ==============================
# SMH / QQQ
# ==============================
for name, ticker in stock_tickers.items():

    print(f"{name} のデータを取得中...")

    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False
    )

    # yfinanceの列がMultiIndexの場合に対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # SMA
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA150"] = df["Close"].rolling(150).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    # RSI
    df["RSI14"] = calculate_rsi(df["Close"], 14)

    # MACD
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = \
        calculate_macd(df["Close"])

    # ADX / ATR
    df["ADX14"], df["ATR14"] = \
        calculate_adx_atr(df, 14)

    filename = f"{name}_technical.csv"

    df.to_csv(filename)

    print(f"{filename} を保存しました。")


# ==============================
# VIX / VIX3M
# ==============================
for name, ticker in volatility_tickers.items():

    print(f"{name} のデータを取得中...")

    df = yf.download(
        ticker,
        period="2y",
        interval="1d",
        auto_adjust=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    filename = f"{name}_technical.csv"

    df.to_csv(filename)

    print(f"{filename} を保存しました。")


print("================================")
print("すべてのデータ処理が完了しました。")
print("================================")
