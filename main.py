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
# 各種ETF・指数
# ============================================================

TICKERS = {
    "SMH": "SMH",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "GLD": "GLD",
    "TLT": "TLT",
    "IEF": "IEF",
    "TIP": "TIP",
    "XLU": "XLU",
    "XLP": "XLP",
    "BIL": "BIL",
    "SGOV": "SGOV",
    "XLK": "XLK",
    "XLE": "XLE",
    "XLF": "XLF",
    "XLV": "XLV",
    "XLI": "XLI",
    "XLB": "XLB",
    "XLY": "XLY",
    "XLRE": "XLRE",
    "XLC": "XLC"
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
        print(f"警告: {name} のデータを取得できませんでした。スキップします。")
        continue

    # MultiIndex対応
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [
        str(column)
        for column in df.columns
    ]

    df = df.sort_index()
    
    # データの最新日付を基準日時として追加
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

# データの最新日付を基準日時として追加
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
    
    # データの最新日付を基準日時として追加
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
# ★追加データ（2026-09 追加）
#
#   既存の出力（*_historical.csv / *_technical.csv / VIX3M / FRED / ALLtec.txt）は
#   一切変えない。ここで取得するのは検証用の新データのみで、
#   ・取得に失敗しても警告を出して続行する（既存パイプラインを止めない）
#   ・結果は EXTRA_DATA_STATUS.txt に一覧で残す
# ============================================================

import os

EXTRA_STATUS = []


def _extra_log(name, ok, detail):
    mark = "OK " if ok else "NG "
    EXTRA_STATUS.append(f"{mark} {name:24s} {detail}")
    print(f"[追加データ] {mark} {name}: {detail}")


def _download_yf(ticker):
    """既存ループと同じ条件で yfinance から日足を取得して整形する。"""

    df = yf.download(
        ticker,
        start=START_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [
        str(column)
        for column in df.columns
    ]

    df = df.sort_index()

    return df


def _save_with_stamp(df, filename):
    df = df.copy()
    df["DataCollectedAt"] = df.index.max().strftime("%Y-%m-%d")
    df.to_csv(filename, index=True)
    return df


# ------------------------------------------------------------
# (1) Yahoo Finance: ドル・商品・市場の幅・半導体指数
# ------------------------------------------------------------

EXTRA_TICKERS = {
    "DXY": "DX-Y.NYB",        # ドル指数
    "COPPER": "HG=F",         # 銅先物（銅/金比率用）
    "GOLD_FUT": "GC=F",       # 金先物（銅/金比率用）
    "SOX": "^SOX",            # フィラデルフィア半導体指数
    "RSP": "RSP",             # S&P500 等ウェイト（市場の幅）
    "SPY": "SPY",             # S&P500 時価総額加重（市場の幅）
}

print("")
print("=" * 70)
print("★追加データ: Yahoo Finance を取得中...")
print("=" * 70)

for name, ticker in EXTRA_TICKERS.items():

    try:
        df = _download_yf(ticker)

        if df is None:
            _extra_log(name, False, f"{ticker} のデータが空")
            continue

        df = _save_with_stamp(df, f"{name}_historical.csv")

        _extra_log(
            name, True,
            f"{df.index.min().date()}〜{df.index.max().date()} {len(df)}件"
        )

    except Exception as e:
        _extra_log(name, False, f"{ticker} 取得失敗: {e!r}")


# ------------------------------------------------------------
# (2) 半導体の主要構成銘柄と「幅（breadth）」
#   ※現在の構成銘柄で過去を見るため生存者バイアスがある点に注意
# ------------------------------------------------------------

SEMI_CONSTITUENTS = [
    "NVDA", "TSM", "AVGO", "ASML", "AMD", "QCOM", "TXN", "INTC",
    "MU", "AMAT", "LRCX", "KLAC", "ADI", "MRVL", "NXPI", "MCHP",
    "ON", "CDNS", "SNPS", "MPWR",
]

SEMI_DIR = "semi_constituents"

print("")
print("=" * 70)
print("★追加データ: 半導体の構成銘柄を取得中...")
print("=" * 70)

semi_close = {}

try:
    os.makedirs(SEMI_DIR, exist_ok=True)
except Exception as e:
    _extra_log("SEMI_DIR", False, f"フォルダ作成失敗: {e!r}")

for symbol in SEMI_CONSTITUENTS:

    try:
        df = _download_yf(symbol)

        if df is None or "Close" not in df.columns:
            _extra_log(f"SEMI_{symbol}", False, "データが空")
            continue

        _save_with_stamp(df, os.path.join(SEMI_DIR, f"{symbol}_historical.csv"))

        semi_close[symbol] = pd.to_numeric(df["Close"], errors="coerce")

        _extra_log(
            f"SEMI_{symbol}", True,
            f"{df.index.min().date()}〜{df.index.max().date()} {len(df)}件"
        )

    except Exception as e:
        _extra_log(f"SEMI_{symbol}", False, f"取得失敗: {e!r}")


try:
    if "SMH" not in all_data:
        raise RuntimeError("SMH のデータが無いため幅を計算できません")

    if len(semi_close) == 0:
        raise RuntimeError("構成銘柄を1つも取得できませんでした")

    # 取引日カレンダーは SMH に合わせる（前方補完はしない）
    calendar = all_data["SMH"].index

    closes = pd.DataFrame(semi_close).reindex(calendar)

    sma50 = closes.rolling(50, min_periods=50).mean()
    sma200 = closes.rolling(200, min_periods=200).mean()
    ret1 = closes / closes.shift(1) - 1      # 前方補完なし（pandasのバージョンに依存しない書き方）

    have50 = closes.notna() & sma50.notna()
    have200 = closes.notna() & sma200.notna()
    have1 = ret1.notna()

    breadth = pd.DataFrame(index=calendar)
    breadth.index.name = "Date"
    breadth["N_Available"] = closes.notna().sum(axis=1)
    breadth["N_For50"] = have50.sum(axis=1)
    breadth["PctAbove50"] = (
        ((closes > sma50) & have50).sum(axis=1)
        / breadth["N_For50"].replace(0, np.nan) * 100
    )
    breadth["N_For200"] = have200.sum(axis=1)
    breadth["PctAbove200"] = (
        ((closes > sma200) & have200).sum(axis=1)
        / breadth["N_For200"].replace(0, np.nan) * 100
    )
    breadth["PctUpDay"] = (
        ((ret1 > 0) & have1).sum(axis=1)
        / have1.sum(axis=1).replace(0, np.nan) * 100
    )
    breadth["MedianRet1Pct"] = ret1.median(axis=1) * 100

    breadth = breadth[breadth["N_Available"] > 0]

    breadth = _save_with_stamp(breadth, "SEMI_BREADTH_historical.csv")

    _extra_log(
        "SEMI_BREADTH", True,
        f"{breadth.index.min().date()}〜{breadth.index.max().date()} "
        f"{len(breadth)}件（銘柄数 {len(semi_close)}）"
    )

except Exception as e:
    _extra_log("SEMI_BREADTH", False, f"計算失敗: {e!r}")


# ------------------------------------------------------------
# (3) CBOE公式データ: VVIX / SKEW / VIX9D
#   ※VIX3Mと同じURL形式。列が「DATE,OPEN,HIGH,LOW,CLOSE」の形式と
#     「DATE,<指数名>」の1列形式の両方に対応する
# ------------------------------------------------------------

CBOE_EXTRA = {
    "VVIX": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VVIX_History.csv",
    "SKEW": "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
    "VIX9D": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
}

print("")
print("=" * 70)
print("★追加データ: CBOE公式データを取得中...")
print("=" * 70)

for name, url in CBOE_EXTRA.items():

    try:
        df = pd.read_csv(url)

        df.columns = [
            str(column).strip().upper()
            for column in df.columns
        ]

        print(f"{name} 取得列: {list(df.columns)}")

        if "DATE" not in df.columns:
            raise RuntimeError(f"DATE列がありません: {list(df.columns)}")

        if "CLOSE" not in df.columns:
            others = [c for c in df.columns if c != "DATE"]
            if name.upper() in df.columns:
                df = df.rename(columns={name.upper(): "CLOSE"})
            elif len(others) == 1:
                df = df.rename(columns={others[0]: "CLOSE"})
            else:
                raise RuntimeError(f"終値の列を特定できません: {list(df.columns)}")

        df["DATE"] = pd.to_datetime(df["DATE"])

        df = df.sort_values("DATE").set_index("DATE")

        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["CLOSE"])

        if df.empty:
            raise RuntimeError("有効な行がありません")

        df = _save_with_stamp(df, f"{name}_historical.csv")

        _extra_log(
            name, True,
            f"{df.index.min().date()}〜{df.index.max().date()} {len(df)}件"
        )

    except Exception as e:
        _extra_log(name, False, f"取得失敗: {e!r}")


# ------------------------------------------------------------
# (4) FRED: 信用スプレッド・ドル・金融環境
#   ※ICE BofA系（ハイイールドOAS等）は2026年4月から直近3年分のみの提供に
#     なったため採用せず、ムーディーズ系（BAA10Y/AAA10Y）を使う
#   ※NFCI は週次・公表に遅れがあるため、検証時は公表日基準で扱うこと
# ------------------------------------------------------------

FRED_EXTRA = {
    "BAA10Y": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA10Y",
        "column": "BAA10Y"
    },
    "AAA10Y": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=AAA10Y",
        "column": "AAA10Y"
    },
    "DollarBroad": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS",
        "column": "DTWEXBGS"
    },
    "NFCI": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=NFCI",
        "column": "NFCI"
    },
}

print("")
print("=" * 70)
print("★追加データ: FREDを取得中...")
print("=" * 70)

for name, info in FRED_EXTRA.items():

    try:
        df = pd.read_csv(info["url"])

        if "observation_date" not in df.columns:
            raise RuntimeError(f"observation_date 列がありません: {list(df.columns)}")

        if info["column"] not in df.columns:
            raise RuntimeError(f"{info['column']} 列がありません: {list(df.columns)}")

        df["observation_date"] = pd.to_datetime(df["observation_date"])

        df = df[["observation_date", info["column"]]].rename(
            columns={
                "observation_date": "Date",
                info["column"]: "Value"
            }
        )

        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        df = df.dropna(subset=["Value"]).sort_values("Date").set_index("Date")

        if df.empty:
            raise RuntimeError("有効な行がありません")

        df = _save_with_stamp(df, f"{name}_historical.csv")

        _extra_log(
            name, True,
            f"{df.index.min().date()}〜{df.index.max().date()} {len(df)}件"
        )

    except Exception as e:
        _extra_log(name, False, f"取得失敗: {e!r}")


# ------------------------------------------------------------
# 追加データの取得結果
# ------------------------------------------------------------

try:
    with open("EXTRA_DATA_STATUS.txt", "w", encoding="utf-8") as f:
        f.write("追加データの取得結果（OK=成功 / NG=失敗。NGでも既存データには影響なし）\n")
        f.write("\n".join(EXTRA_STATUS))
        f.write("\n")
except Exception as e:
    print(f"[追加データ] EXTRA_DATA_STATUS.txt の保存に失敗: {e!r}")


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
    "GLD",
    "TLT",
    "IEF",
    "TIP",
    "XLU",
    "XLP",
    "BIL",
    "SGOV",
    "XLK",
    "XLE",
    "XLF",
    "XLV",
    "XLI",
    "XLB",
    "XLY",
    "XLRE",
    "XLC",
    "VIX3M",
    "10Y_Treasury",
    "2Y_Treasury",
    "FedFunds",
    "CPI"
]


for name in ALLTEC_DATASETS:
    
    if name not in all_data:
        continue

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
    "取得完了ETF・指数:"
)
print(
    "SMH / QQQ / VIX / GLD / TLT / IEF / TIP / XLU / XLP / BIL / SGOV"
)
print(
    "XLK / XLE / XLF / XLV / XLI / XLB / XLY / XLRE / XLC / VIX3M"
)

print("")
print(
    "取得完了マクロデータ:"
)
print(
    "10Y Treasury / 2Y Treasury / FedFunds / CPI"
)

print("")
print(
    "★追加データ（検証用、失敗しても既存データに影響なし）:"
)
print(
    "DXY / COPPER / GOLD_FUT / SOX / RSP / SPY / 半導体構成銘柄20 / SEMI_BREADTH"
)
print(
    "VVIX / SKEW / VIX9D / BAA10Y / AAA10Y / DollarBroad / NFCI"
)
print(
    "取得結果の一覧: EXTRA_DATA_STATUS.txt"
)

print("")
print("すべての処理が完了しました。")
