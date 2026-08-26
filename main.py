import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# 設定
# ============================================================

START_DATE = "1995-01-01"
VIX3M_START_DATE = "2007-12-04"
ALLTEC_DAYS = 60

# CBOE公式 VIX3M
VIX3M_URL = (
    "https://cdn.cboe.com/api/global/us_indices/"
    "daily_prices/VIX3M_History.csv"
)

# FRED
FRED_DATA = {
    "10Y_Treasury": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
        "column": "DGS10"
    },
    "2Y_Treasury": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2",
        "column": "DGS2"
    },
    "FedFunds": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS",
        "column": "FEDFUNDS"
    },
    "CPI": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL",
        "column": "CPIAUCSL"
    }
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
# データ格納
# ============================================================

all_data = {}


# ============================================================
# Yahoo Finance
# SMH / QQQ / VIX
# ============================================================

TICKERS = {
    "SMH": "SMH",
    "QQQ": "QQQ",
    "VIX": "^VIX"
}


for name, ticker in TICKERS.items():

    print("")
    print("=" * 70)
    print(f"{name} ({ticker}) のデータを取得中...")
    print("=" * 70)

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

    # MultiIndex対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [
        str(column)
        for column in df.columns
    ]

    df = df.sort_index()
    
    # 【修正】データの最新日付を基準日時として追加
    df["DataCollectedAt"] = df.index.max().strftime("%Y-%m-%d")

    # --------------------------------------------------------
    # ヒストリカルデータ
    # 元データを削らず保存
    # --------------------------------------------------------

    historical_filename = (
        f"{name}_historical.csv"
    )

    df.to_csv(
        historical_filename,
        index=True
    )

    print(
        f"{historical_filename} 保存完了"
    )

    print(
        f"最古: {df.index.min()}"
    )

    print(
        f"最新: {df.index.max()}"
    )

    print(
        f"件数: {len(df)}"
    )

    # --------------------------------------------------------
    # テクニカル
    # --------------------------------------------------------

    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA100"] = df["Close"].rolling(100).mean()
    df["SMA150"] = df["Close"].rolling(150).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    df["RSI14"] = calculate_rsi(
        df["Close"],
        14
    )

    (
        df["MACD"],
        df["MACD_Signal"],
        df["MACD_Hist"]
    ) = calculate_macd(
        df["Close"]
    )

    df["ADX14"], df["ATR14"] = calculate_adx_atr(
        df,
        14
    )

    technical_filename = (
        f"{name}_technical.csv"
    )

    df.to_csv(
        technical_filename,
        index=True
    )

    print(
        f"{technical_filename} 保存完了"
    )

    all_data[name] = df


# ============================================================
# VIX3M
# CBOE公式データ
# ============================================================

print("")
print("=" * 70)
print("VIX3M (CBOE公式データ) を取得中...")
print("=" * 70)

vix3m = pd.read_csv(
    VIX3M_URL
)

vix3m.columns = [
    str(column).strip()
    for column in vix3m.columns
]

print(
    "VIX3M取得列:",
    list(vix3m.columns)
)

if "DATE" not in vix3m.columns:

    raise RuntimeError(
        "VIX3MデータにDATE列がありません。"
    )

vix3m["DATE"] = pd.to_datetime(
    vix3m["DATE"]
)

vix3m = vix3m[
    vix3m["DATE"] >= VIX3M_START_DATE
]

vix3m = vix3m.sort_values(
    "DATE"
)

vix3m = vix3m.set_index(
    "DATE"
)

if "CLOSE" not in vix3m.columns:

    raise RuntimeError(
        "VIX3MデータにCLOSE列がありません。"
    )

for column in vix3m.columns:

    vix3m[column] = pd.to_numeric(
        vix3m[column],
        errors="coerce"
    )

# 【修正】データの最新日付を基準日時として追加
vix3m["DataCollectedAt"] = vix3m.index.max().strftime("%Y-%m-%d")

# ------------------------------------------------------------
# VIX3M ヒストリカル
# ------------------------------------------------------------

vix3m.to_csv(
    "VIX3M_historical.csv",
    index=True
)

print(
    "VIX3M_historical.csv 保存完了"
)

print(
    f"最古: {vix3m.index.min()}"
)

print(
    f"最新: {vix3m.index.max()}"
)

print(
    f"件数: {len(vix3m)}"
)


# ------------------------------------------------------------
# VIX3M テクニカル
# ------------------------------------------------------------

vix3m_technical = vix3m.copy()

close = vix3m_technical["CLOSE"]

vix3m_technical["SMA20"] = (
    close.rolling(20).mean()
)

vix3m_technical["SMA50"] = (
    close.rolling(50).mean()
)

vix3m_technical["SMA100"] = (
    close.rolling(100).mean()
)

vix3m_technical["SMA150"] = (
    close.rolling(150).mean()
)

vix3m_technical["SMA200"] = (
    close.rolling(200).mean()
)

vix3m_technical["RSI14"] = (
    calculate_rsi(
        close,
        14
    )
)

(
    vix3m_technical["MACD"],
    vix3m_technical["MACD_Signal"],
    vix3m_technical["MACD_Hist"]
) = calculate_macd(
    close
)


# High / Lowがある場合のみADX / ATR
if (
    "HIGH" in vix3m_technical.columns
    and
    "LOW" in vix3m_technical.columns
):

    adx_df = pd.DataFrame(
        {
            "High": vix3m_technical["HIGH"],
            "Low": vix3m_technical["LOW"],
            "Close": vix3m_technical["CLOSE"]
        },
        index=vix3m_technical.index
    )

    (
        vix3m_technical["ADX14"],
        vix3m_technical["ATR14"]
    ) = calculate_adx_atr(
        adx_df,
        14
    )


vix3m_technical.to_csv(
    "VIX3M_technical.csv",
    index=True
)

print(
    "VIX3M_technical.csv 保存完了"
)

all_data["VIX3M"] = vix3m_technical


# ============================================================
# FRED マクロ経済データ
# ============================================================

print("")
print("=" * 70)
print("FREDマクロ経済データを取得します")
print("=" * 70)


for name, info in FRED_DATA.items():

    print("")
    print(
        f"{name} をFREDから取得中..."
    )

    df = pd.read_csv(
        info["url"]
    )

    if "observation_date" not in df.columns:

        raise RuntimeError(
            f"{name} に observation_date 列がありません。"
        )

    df["observation_date"] = pd.to_datetime(
        df["observation_date"]
    )

    df = df[
        [
            "observation_date",
            info["column"]
        ]
    ]

    df = df.rename(
        columns={
            "observation_date": "Date",
            info["column"]: "Value"
        }
    )

    df["Value"] = pd.to_numeric(
        df["Value"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Value"]
    )

    df = df.sort_values(
        "Date"
    )

    df = df.set_index(
        "Date"
    )
    
    # 【修正】データの最新日付を基準日時として追加
    df["DataCollectedAt"] = df.index.max().strftime("%Y-%m-%d")

    # --------------------------------------------------------
    # ヒストリカルデータ
    # 元データを削らず保存
    # --------------------------------------------------------

    historical_filename = (
        f"{name}_historical.csv"
    )

    df.to_csv(
        historical_filename
    )

    print(
        f"{historical_filename} 保存完了"
    )

    print(
        f"最古: {df.index.min()}"
    )

    print(
        f"最新: {df.index.max()}"
    )

    print(
        f"件数: {len(df)}"
    )

    all_data[name] = df


# ============================================================
# ALLtec.txt
# 最新60観測値を収録
# ============================================================

print("")
print("=" * 70)
print("ALLtec.txtを作成しています...")
print("=" * 70)


output = []

output.append("=" * 80)
output.append("AI TECHNICAL MARKET DATA")
output.append("=" * 80)
output.append("")
output.append(
    "Source: Yahoo Finance + CBOE + FRED"
)
output.append(
    "Historical data: oldest available from requested start date"
)
output.append(
    f"AI analysis period: latest {ALLTEC_DAYS} observations"
)
output.append(
    "Historical and technical columns are preserved."
)
output.append("")


# ============================================================
# ALLtec対象データ
# ============================================================

ALLTEC_DATASETS = [
    "SMH",
    "QQQ",
    "VIX",
    "VIX3M",
    "10Y_Treasury",
    "2Y_Treasury",
    "FedFunds",
    "CPI"
]


for name in ALLTEC_DATASETS:

    df = all_data[name].tail(
        ALLTEC_DAYS
    ).copy()

    output.append("")
    output.append("=" * 80)
    output.append(name)
    output.append("=" * 80)
    output.append("")

    columns = list(df.columns)

    output.append(
        "Date," + ",".join(columns)
    )

    for date, row in df.iterrows():

        values = []

        values.append(
            date.strftime("%Y-%m-%d")
        )

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
print("=" * 70)
print("ALLtec.txt 保存完了")
print("=" * 70)

print("")
print(
    f"各データの最新{ALLTEC_DAYS}観測値を収録"
)

print("")
print(
    "SMH / QQQ / VIX / VIX3M"
)

print(
    "10Y Treasury / 2Y Treasury / FedFunds / CPI"
)

print("")
print("すべての処理が完了しました。")
