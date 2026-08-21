import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# 設定
# ============================================================

START_DATE = "1995-01-01"
ALLTEC_DAYS = 60

TICKERS = {
    "SMH": "SMH",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M"
}


# ============================================================
# RSI
# ============================================================

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


# ============================================================
# MACD
# ============================================================

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


# ============================================================
# ADX / ATR
# ============================================================

def calculate_adx_atr(df, period=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

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

    atr = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            min_periods=period,
            adjust=False
        ).mean()
        / atr
    )

    denominator = plus_di + minus_di

    dx = (
        100
        * (plus_di - minus_di).abs()
        / denominator.replace(0, np.nan)
    )

    adx = dx.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    return adx, atr


# ============================================================
# データ取得
# ============================================================

all_data = {}

for name, ticker in TICKERS.items():

    print("")
    print("=" * 60)
    print(f"{name} ({ticker}) のデータを取得中...")
    print("=" * 60)

    df = yf.download(
        ticker,
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError(
            f"{name} のデータを取得できませんでした。"
        )

    # yfinanceがMultiIndexを返す場合に対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 念のため列名を文字列化
    df.columns = [str(col) for col in df.columns]

    # 日付順に並べる
    df = df.sort_index()

    # --------------------------------------------------------
    # ヒストリカルデータを完全保存
    # --------------------------------------------------------

    historical_filename = f"{name}_historical.csv"

    df.to_csv(
        historical_filename,
        index=True
    )

    print(
        f"{historical_filename} 保存完了"
    )

    print(
        f"期間: {df.index.min().date()} "
        f"～ {df.index.max().date()}"
    )

    print(
        f"件数: {len(df)}"
    )

    # --------------------------------------------------------
    # テクニカル指標を追加
    # 元データの列は削除しない
    # --------------------------------------------------------

    # SMA
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA150"] = df["Close"].rolling(150).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    # VIX / VIX3M用の短期SMA
    if name in ["VIX", "VIX3M"]:
        df["SMA20"] = df["Close"].rolling(20).mean()

    # RSI
    df["RSI14"] = calculate_rsi(
        df["Close"],
        14
    )

    # MACD
    (
        df["MACD"],
        df["MACD_Signal"],
        df["MACD_Hist"]
    ) = calculate_macd(
        df["Close"]
    )

    # ADX / ATR
    df["ADX14"], df["ATR14"] = calculate_adx_atr(
        df,
        14
    )

    # --------------------------------------------------------
    # テクニカルCSV保存
    # 元データ＋テクニカル指標
    # --------------------------------------------------------

    technical_filename = f"{name}_technical.csv"

    df.to_csv(
        technical_filename,
        index=True
    )

    print(
        f"{technical_filename} 保存完了"
    )

    all_data[name] = df


# ============================================================
# ALLtec.txt作成
# ============================================================

print("")
print("=" * 60)
print("ALLtec.txt を作成しています...")
print("=" * 60)

output = []

output.append("=" * 80)
output.append("AI TECHNICAL MARKET DATA")
output.append("=" * 80)
output.append("")
output.append(
    "Historical data source: Yahoo Finance via yfinance"
)
output.append(
    f"Historical data start request: {START_DATE}"
)
output.append(
    f"AI analysis data: latest {ALLTEC_DAYS} trading days"
)
output.append(
    "Historical and technical columns are preserved."
)
output.append("")


# ============================================================
# 各銘柄について直近60営業日を出力
# ============================================================

for name in TICKERS.keys():

    df = all_data[name].tail(ALLTEC_DAYS).copy()

    output.append("")
    output.append("=" * 80)
    output.append(name)
    output.append("=" * 80)
    output.append("")

    # --------------------------------------------------------
    # 列名
    # --------------------------------------------------------

    columns = list(df.columns)

    # Dateを先頭にする
    output.append(
        "Date," + ",".join(columns)
    )

    # --------------------------------------------------------
    # データ
    # --------------------------------------------------------

    for date, row in df.iterrows():

        values = []

        # 日付
        values.append(
            date.strftime("%Y-%m-%d")
        )

        # 全列をそのまま出力
        for column in columns:

            value = row[column]

            if pd.isna(value):
                values.append("")
            else:
                try:
                    values.append(
                        f"{float(value):.6f}"
                    )
                except:
                    values.append(
                        str(value)
                    )

        output.append(
            ",".join(values)
        )


# ============================================================
# ALLtec.txt保存
# ============================================================

with open(
    "ALLtec.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(output)
    )


print("")
print("=" * 60)
print("ALLtec.txt 保存完了")
print("=" * 60)

print("")
print("処理対象:")
print("SMH")
print("QQQ")
print("VIX")
print("VIX3M")

print("")
print(
    f"ALLtec.txt: 直近{ALLTEC_DAYS}営業日"
)

print("")
print("すべての処理が完了しました。")
