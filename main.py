import time

import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# 設定
# ============================================================

START_DATE = "1995-01-01"
# 2026-10: ^SOX（フィラデルフィア半導体指数）だけは、Yahooにある最古の日（1994-05-04）から取る。
#          1990年を指定しておけば、Yahooが持つ最も古い日から返ってくる（それより前の分は存在しない）。
#          SOX以外の銘柄の開始日（START_DATE）は従来のまま。
SOX_START_DATE = "1990-01-01"
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

    # 2026-10: Yahooが一時的に空を返すことがあるため、最大3回まで試す
    #          （取得できたときの出力は従来と同じ。失敗が続いたときの扱いも従来と同じ）
    df = None

    for _attempt in range(3):

        df = yf.download(
            ticker,
            start=START_DATE,
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False
        )

        if df is not None and not df.empty:
            break

        if _attempt < 2:
            time.sleep(3)

    if df is None or df.empty:
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


def _download_yf(ticker, start=None):
    """既存ループと同じ条件で yfinance から日足を取得して整形する。
    start を省略すると従来どおり START_DATE から取る（2026-10: 銘柄ごとに開始日を変えられるよう引数を追加）。"""

    df = yf.download(
        ticker,
        start=(start if start is not None else START_DATE),
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
    # ---- 2026-10 追加（情報として使う系列。売買対象ではない）----
    "MOVE": "^MOVE",          # 債券のボラティリティ（TLT/IEF退避の判定用）
    "HYG": "HYG",             # ハイイールド債（信用スプレッドの代理。BAA10Yより日次で速い）
    "LQD": "LQD",             # 投資適格社債（HYGとの比で信用の質の変化を見る）
    "NQ_FUT": "NQ=F",         # ナスダック100先物（引け後〜翌寄付き前の動きの確認用）
    "ES_FUT": "ES=F",         # S&P500先物（同上）
}

print("")
print("=" * 70)
print("★追加データ: Yahoo Finance を取得中...")
print("=" * 70)

# 銘柄ごとの開始日（ここに無い銘柄は START_DATE）
EXTRA_START = {
    "SOX": SOX_START_DATE,
}

for name, ticker in EXTRA_TICKERS.items():

    try:
        _start = EXTRA_START.get(name)

        df = _download_yf(ticker, start=_start)

        # 開始日を変えた銘柄は、Yahooが一時的に空を返したときに1回だけ取り直す
        if df is None and _start is not None:
            time.sleep(3)
            df = _download_yf(ticker, start=_start)

        if df is None:
            _extra_log(name, False, f"{ticker} のデータが空")
            continue

        df = _save_with_stamp(df, f"{name}_historical.csv")

        _detail = f"{df.index.min().date()}〜{df.index.max().date()} {len(df)}件"

        # SOX: 1994年分が入っているか（Yahooの最古は1994-05-04のはず）を一覧に残す
        if name == "SOX":
            if df.index.min() <= pd.Timestamp("1994-12-31"):
                _detail += "（1994年分を含む）"
            else:
                _detail += "（※1994年分なし。Yahoo側の提供開始が想定より遅い可能性）"

        _extra_log(name, True, _detail)

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
    # ---- 2026-10 追加 ----
    "VXN": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv",      # ナスダック100のVIX（2001〜）
    "VIX6M": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX6M_History.csv",  # 6か月VIX（期間構造の長い側）
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
    # ---- 2026-10 追加 ----
    "Treasury3M": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO",
        "column": "DGS3MO"          # 3か月国債利回り（現金退避の利回りの代理・逆イールド判定）
    },
    "T10Y3M": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M",
        "column": "T10Y3M"          # 10年−3か月（景気後退予測に最もよく使われる傾き）
    },
    "STLFSI4": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=STLFSI4",
        "column": "STLFSI4"         # セントルイス連銀の金融ストレス指数（週次）
    },
    "RealYield10Y": {
        "url": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10",
        "column": "DFII10"          # 10年実質利回り（TLTの下落要因の切り分け）
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


# ============================================================
# ★追加データ（2026-10 追加その2）
#
#   ここから先も「既存の出力は変えない／失敗しても止めない」方針は同じ。
#   目的は、今の手元データでは検証できない3つの方向を検証できるようにすること。
#     (A) アジア時間の半導体株  … 米国の引け後〜翌寄付き前に確定する情報
#     (B) 防御系・保有先の候補ETF … 退避先・保有先の候補比較（※売買できるかは別途確認）
#     (C) SMH/QQQ/TLT/IEF/VIX の分足の蓄積 … 「引け前の判定」の誤差を実測するため
#   ※ (A) の日付の対応: アジアの d 日の終値は、米国の d 日の寄付きより前に確定する
#     （米国 d−1 日の引け後〜d 日の寄付き前の動き）。翌営業日の寄付きで執行する夜の判断材料になる
#   ※ (A) は情報として使うだけで売買対象ではない。(B) のうち SBI米国株アプリで
#     売買できない銘柄は、手法には組み込まない（データだけ取っておく）
# ============================================================

import datetime as _dt


def _download_yf_retry(ticker, tries=2, wait=3):
    """_download_yf を一時的な失敗に備えて2回まで試す。日足の索引がタイムゾーン付きなら外す。"""

    last = None

    for i in range(tries):
        try:
            df = _download_yf(ticker)
            if df is not None:
                if getattr(df.index, "tz", None) is not None:
                    df.index = df.index.tz_localize(None)
                return df
        except Exception as e:
            last = e
        if i < tries - 1:
            time.sleep(wait)

    if last is not None:
        raise last

    return None


def _fetch_group(tag, tickers, folder):
    """tickers={保存名: Yahooの銘柄} を folder に *_historical.csv で保存し、終値の Series を返す。"""

    closes = {}

    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        _extra_log(f"{tag}_DIR", False, f"フォルダ作成失敗: {e!r}")
        return closes

    for name, ticker in tickers.items():

        try:
            df = _download_yf_retry(ticker)

            if df is None or "Close" not in df.columns:
                _extra_log(f"{tag}_{name}", False, f"{ticker} のデータが空")
                continue

            df = _save_with_stamp(df, os.path.join(folder, f"{name}_historical.csv"))

            closes[name] = pd.to_numeric(df["Close"], errors="coerce")

            _extra_log(
                f"{tag}_{name}", True,
                f"{ticker} {df.index.min().date()}〜{df.index.max().date()} {len(df)}件"
            )

        except Exception as e:
            _extra_log(f"{tag}_{name}", False, f"{ticker} 取得失敗: {e!r}")

    return closes


# ------------------------------------------------------------
# (5) アジア・欧州の半導体株と指数（asia_session/）
# ------------------------------------------------------------

ASIA_DIR = "asia_session"

ASIA_TICKERS = {
    "8035_T": "8035.T",          # 東京エレクトロン
    "6857_T": "6857.T",          # アドバンテスト
    "6920_T": "6920.T",          # レーザーテック
    "6146_T": "6146.T",          # ディスコ
    "2330_TW": "2330.TW",        # TSMC（台湾本土）
    "000660_KS": "000660.KS",    # SKハイニックス
    "005930_KS": "005930.KS",    # サムスン電子
    "N225": "^N225",             # 日経平均
    "TWII": "^TWII",             # 台湾加権
    "KS11": "^KS11",             # 韓国KOSPI
    "ASML_AS": "ASML.AS",        # ASML（アムステルダム。米国の引けの前に閉まる）
    "IFX_DE": "IFX.DE",          # インフィニオン（フランクフルト。同上）
}

# 日次バスケットに使う銘柄（アジアの半導体7銘柄。指数と欧州は含めない）
ASIA_BASKET = ["8035_T", "6857_T", "6920_T", "6146_T", "2330_TW", "000660_KS", "005930_KS"]

print("")
print("=" * 70)
print("★追加データ: アジア・欧州の半導体株と指数を取得中...")
print("=" * 70)

asia_close = _fetch_group("ASIA", ASIA_TICKERS, ASIA_DIR)

try:
    names = [n for n in ASIA_BASKET if n in asia_close]

    if len(names) == 0:
        raise RuntimeError("バスケットの銘柄を1つも取得できませんでした")

    if "SMH" not in all_data:
        raise RuntimeError("SMH のデータが無いためバスケットを作れません")

    rets = {}

    for n in names:
        s = asia_close[n].dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        rets[n] = s.pct_change()       # 各銘柄自身の取引日どうしの騰落率（休場をまたぐ日は複数日分）

    # 米国（SMH）の営業日に合わせる。前方補完はしない（その日に取引が無い銘柄は除いて平均する）
    calendar = all_data["SMH"].index

    R = pd.DataFrame(rets).reindex(calendar)

    basket = pd.DataFrame(index=calendar)
    basket.index.name = "Date"
    basket["N_Available"] = R.notna().sum(axis=1)
    basket["AsiaSemiRetPct"] = R.mean(axis=1) * 100       # 等ウェイトの平均騰落率（%）
    basket = basket[basket["N_Available"] > 0]

    basket = _save_with_stamp(basket, "ASIA_SEMI_BASKET_historical.csv")

    _extra_log(
        "ASIA_SEMI_BASKET", True,
        f"{basket.index.min().date()}〜{basket.index.max().date()} "
        f"{len(basket)}件（銘柄数 {len(names)}）"
    )

except Exception as e:
    _extra_log("ASIA_SEMI_BASKET", False, f"計算失敗: {e!r}")


# ------------------------------------------------------------
# (6) 防御系・保有先の候補ETF（candidates/）
#   ※ SBI米国株アプリで売買できるかは各自で確認すること。
#     売買できない銘柄は手法に組み込まない（データだけ残す）。
#   BTAL … 低ベータ買い・高ベータ売りの市場中立（株安時に上がりやすい）
#   DBMF/KMLM/CTA … 先物のトレンドフォロー（株債同時安の局面の候補）
#   TAIL … 株価指数のプット保有（暴落保険）
#   SOXX/SOXQ/XSD/QQQM … SMH・QQQの代わりに持つ場合の候補（経費率・構成の違い）
# ------------------------------------------------------------

CANDIDATE_DIR = "candidates"

CANDIDATE_TICKERS = {
    "BTAL": "BTAL",
    "DBMF": "DBMF",
    "KMLM": "KMLM",
    "CTA": "CTA",
    "TAIL": "TAIL",
    "SOXX": "SOXX",
    "SOXQ": "SOXQ",
    "XSD": "XSD",
    "QQQM": "QQQM",
}

print("")
print("=" * 70)
print("★追加データ: 防御系・保有先の候補ETFを取得中...")
print("=" * 70)

_fetch_group("CAND", CANDIDATE_TICKERS, CANDIDATE_DIR)


# ------------------------------------------------------------
# (7) 分足の蓄積（intraday/）
#   Yahooの分足は 5分足=直近約60日・1時間足=直近約730日しか遡れない。
#   取り逃すと二度と取れないため、毎回の取得分を既存ファイルに足して残す。
#   5分足は60日分しか入らないので、遅くとも8週間に1回は実行すること。
#   用途: 引け前（15:30〜15:55）の価格と最終の終値の差を実測し、
#         「引けで執行してよいか」の判定の誤差（今は0.3%と仮定）を置き換える。
#   *_preclose_historical.csv は 5分足から日ごとに作る要約（毎回作り直す）。
# ------------------------------------------------------------

INTRADAY_ENABLED = True

INTRADAY_DIR = "intraday"
INTRADAY_TZ = "America/New_York"

INTRADAY_TICKERS = {
    "SMH": "SMH",
    "QQQ": "QQQ",
    "TLT": "TLT",
    "IEF": "IEF",
    "VIX": "^VIX",
}

INTRADAY_SPECS = [
    ("5m", "58d"),     # 上限は60日。境界での失敗を避けて58日
    ("1h", "720d"),    # 上限は730日
]


def _download_intraday(ticker, interval, period):
    """分足を取得し、索引をUTCに揃えて返す。"""

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        progress=False
    )

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [str(column) for column in df.columns]

    idx = pd.DatetimeIndex(df.index)

    if idx.tz is None:
        # タイムゾーンが付いていない場合は取引所の現地時間とみなす
        idx = idx.tz_localize(INTRADAY_TZ, ambiguous="NaT", nonexistent="NaT")

    df.index = idx.tz_convert("UTC")
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()

    return df


def _merge_intraday(path, new):
    """既存ファイルに今回分を足して、重複は今回分を優先し、時刻順に並べる。
    既存ファイルが読めないときは例外にして、上書きしない。"""

    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0)
        old.index = pd.to_datetime(old.index, utc=True)
        old = old.drop(columns=[c for c in ["DataCollectedAt"] if c in old.columns])
        merged = pd.concat([old, new])
    else:
        merged = new.copy()

    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    out = merged.copy()
    out.index = out.index.tz_convert(INTRADAY_TZ)
    out.index.name = "Datetime"
    out["DataCollectedAt"] = out.index.max().strftime("%Y-%m-%d")
    out.to_csv(path, index=True)

    return merged


def _preclose_summary(df5):
    """5分足から、日ごとの「引け前の価格」の要約を作る。
    P1530 = 15:30時点の価格（15:25開始の足の終値）。P1600 = 最後の足の終値。"""

    d = df5[["Open", "High", "Low", "Close"]].apply(pd.to_numeric, errors="coerce")
    d = d.tz_convert(INTRADAY_TZ)

    def px(g, hh, mm):
        sel = g[g.index.time < _dt.time(hh, mm)]
        return sel["Close"].iloc[-1] if len(sel) else np.nan

    rows = []

    for day, g in d.groupby(d.index.date):

        g = g.sort_index()

        rows.append({
            "Date": pd.Timestamp(day),
            "NBars": len(g),
            "FirstBar": g.index[0].strftime("%H:%M"),
            "LastBar": g.index[-1].strftime("%H:%M"),    # 15:55 以外なら半日取引か途中の日
            "Open930": g["Open"].iloc[0],
            "P1500": px(g, 15, 0),
            "P1530": px(g, 15, 30),
            "P1550": px(g, 15, 50),
            "P1555": px(g, 15, 55),
            "P1600": g["Close"].iloc[-1],
            "SessionHigh": g["High"].max(),
            "SessionLow": g["Low"].min(),
        })

    return pd.DataFrame(rows).set_index("Date")


if INTRADAY_ENABLED:

    print("")
    print("=" * 70)
    print("★追加データ: 分足を取得して蓄積中...")
    print("=" * 70)

    try:
        os.makedirs(INTRADAY_DIR, exist_ok=True)
        _intraday_ok = True
    except Exception as e:
        _intraday_ok = False
        _extra_log("INTRADAY_DIR", False, f"フォルダ作成失敗: {e!r}")

    if _intraday_ok:

        for name, ticker in INTRADAY_TICKERS.items():

            for interval, period in INTRADAY_SPECS:

                tag = f"INTRA_{name}_{interval}"
                path = os.path.join(INTRADAY_DIR, f"{name}_{interval}.csv")

                try:
                    new = None

                    for attempt in range(2):
                        try:
                            new = _download_intraday(ticker, interval, period)
                        except Exception:
                            new = None
                            if attempt == 1:
                                raise
                        if new is not None:
                            break
                        time.sleep(3)

                    if new is None:
                        _extra_log(tag, False, f"{ticker} の分足が空")
                        continue

                    merged = _merge_intraday(path, new)

                    _extra_log(
                        tag, True,
                        f"{ticker} 今回{len(new)}本 / 累計{len(merged)}本 "
                        f"{merged.index.min().strftime('%Y-%m-%d')}〜"
                        f"{merged.index.max().strftime('%Y-%m-%d')}"
                    )

                    if interval == "5m":

                        summary = _preclose_summary(merged)

                        _save_with_stamp(
                            summary,
                            os.path.join(INTRADAY_DIR, f"{name}_preclose_historical.csv")
                        )

                        _extra_log(
                            f"INTRA_{name}_preclose", True,
                            f"{len(summary)}日分"
                        )

                except Exception as e:
                    _extra_log(tag, False, f"{ticker} 取得失敗: {e!r}")


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
    "MOVE / HYG / LQD / NQ先物 / ES先物 / VXN / VIX6M / Treasury3M / T10Y3M / STLFSI4 / RealYield10Y"
)
print(
    "アジア・欧州の半導体株と指数（asia_session/）/ ASIA_SEMI_BASKET"
)
print(
    "防御系・保有先の候補ETF（candidates/）/ 分足の蓄積（intraday/）"
)
print(
    "取得結果の一覧: EXTRA_DATA_STATUS.txt"
)

print("")
print("すべての処理が完了しました。")
