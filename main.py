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
# VIX3M の場中の暫定行（2026-10、ルールエンジン v8.29 の C2 用）
# ============================================================
# CBOE の VIX3M は引け後に確定するので、場中（引けの数分前）に①を走らせると当日の行がない。
# ルールエンジン v8.29 の C2（VIX の30上抜け × VIX ÷ VIX3M ≧ 1）は当日の VIX3M が必要なので、
# 当日の VIX の行があり、CBOE の最新日がそれより前のときだけ、Yahoo の ^VIX3M（取れなければ ^VXV）の
# 当日の足を『暫定の行』として足す。CBOE の値が出たら、次の実行で CBOE の値に置き換わる（毎回 CBOE の全履歴から作り直す）。
# 取れなかったとき・値が不自然なときは足さない（②③は、その日の C2 を判定しない＝v8.28 と同じ判定）。
# ※ Yahoo のシンボル（^VIX3M／^VXV）は未検証（この環境から Yahoo に接続できないため）。取れなければ何も足さないので安全側。

VIX3M_YAHOO_SYMBOLS = ["^VIX3M", "^VXV"]


def append_vix3m_provisional(vix3m, vix_df, symbols=None, download=None):
    """CBOE の VIX3M（日付を索引にした DataFrame。列は OPEN/HIGH/LOW/CLOSE）に、場中の暫定行を足して返す。
    足す条件: (1) vix_df（Yahoo の ^VIX の日足）の最新日が CBOE の最新日より後、(2) Yahoo の VIX3M の足がその間の日付にある、
    (3) 値が 5〜150 で、同じ日の VIX との比が 0.5〜2 の範囲。足した日付・値を表示する。失敗しても例外は出さない（何も足さない）。"""
    symbols = symbols or VIX3M_YAHOO_SYMBOLS
    download = download or yf.download
    try:
        if vix_df is None or len(vix_df) == 0 or len(vix3m) == 0:
            print("VIX3M暫定行: VIXの当日の行がないため、足しません。")
            return vix3m
        cboe_last = pd.Timestamp(vix3m.index.max()).normalize()
        vix_last = pd.Timestamp(vix_df.index.max()).normalize()
        if vix_last <= cboe_last:
            print(f"VIX3M暫定行: CBOEの最新日（{cboe_last:%Y-%m-%d}）がVIXの最新日（{vix_last:%Y-%m-%d}）以降のため、足しません。")
            return vix3m
        for sym in symbols:
            try:
                d = None
                for _attempt in range(2):
                    d = download(sym, period="10d", interval="1d", auto_adjust=False, actions=False, progress=False)
                    if d is not None and len(d) > 0:
                        break
                    time.sleep(2)
                if d is None or len(d) == 0:
                    print(f"VIX3M暫定行: {sym} を取得できませんでした。")
                    continue
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d.columns = [str(c) for c in d.columns]
                d = d.sort_index()
                d.index = pd.to_datetime(d.index).tz_localize(None).normalize() if getattr(d.index, "tz", None) is not None else pd.to_datetime(d.index).normalize()
                if "Close" not in d.columns:
                    print(f"VIX3M暫定行: {sym} に Close 列がありません。")
                    continue
                new = d[(d.index > cboe_last) & (d.index <= vix_last)].dropna(subset=["Close"])
                rows = []
                for dt, r in new.iterrows():
                    c = float(r["Close"])
                    vx = vix_df["Close"].reindex([dt]).iloc[0] if "Close" in vix_df.columns else float("nan")
                    if not (5.0 <= c <= 150.0):
                        print(f"VIX3M暫定行: {sym} {dt:%Y-%m-%d} の値 {c} が範囲外のため足しません。")
                        continue
                    if pd.notna(vx) and not (0.5 <= c / float(vx) <= 2.0):
                        print(f"VIX3M暫定行: {sym} {dt:%Y-%m-%d} の値 {c:.2f} がVIX {float(vx):.2f} と離れすぎのため足しません。")
                        continue
                    rows.append((dt, {"OPEN": float(r.get("Open", c)), "HIGH": float(r.get("High", c)), "LOW": float(r.get("Low", c)), "CLOSE": c}))
                if not rows:
                    print(f"VIX3M暫定行: {sym} にCBOEの最新日（{cboe_last:%Y-%m-%d}）より後の使える足がありません。")
                    continue
                add = pd.DataFrame([v for _, v in rows], index=pd.DatetimeIndex([k for k, _ in rows], name=vix3m.index.name))
                for col in vix3m.columns:
                    if col not in add.columns:
                        add[col] = float("nan")
                out = pd.concat([vix3m, add[list(vix3m.columns)]]).sort_index()
                out.index.name = vix3m.index.name
                for dt, v in rows:
                    print(f"VIX3M暫定行: {sym} の当日の足を足しました: {dt:%Y-%m-%d} 終値 {v['CLOSE']:.2f}（場中の暫定値。CBOEの確定値が出たら次の実行で置き換わります）")
                return out
            except Exception as e:
                print(f"VIX3M暫定行: {sym} の処理に失敗しました（{e!r}）。")
                continue
        print("VIX3M暫定行: どのシンボルからも取得できなかったため、足しません（当日のC2は判定されません）。")
        return vix3m
    except Exception as e:
        print(f"VIX3M暫定行: 処理に失敗しました（{e!r}）。足しません。")
        return vix3m



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

# 2026-10（v8.29）: 場中の暫定行（CBOEの当日の行がまだ無いとき、Yahooの^VIX3Mの当日の足を足す）
vix3m = append_vix3m_provisional(
    vix3m,
    all_data["VIX"] if "VIX" in all_data else None
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
    # 第49回: 退避先TLTの代わりに持つ候補（TLTより金利感応度が高い／同程度の長期債ETF）。
    #   EDV … バンガード超長期米国債ETF（デュレーション約25年。TLTの値動きの約1.4〜1.5倍）。SBI証券の取扱一覧に掲載あり
    #   ZROZ/GOVZ … ゼロクーポン債（25年超）。SPTL/VGLT … TLTと同程度（比較用）
    "EDV": "EDV",
    "ZROZ": "ZROZ",
    "GOVZ": "GOVZ",
    "SPTL": "SPTL",
    "VGLT": "VGLT",
    # 第51回: MAGS（Roundhill Magnificent Seven ETF）を検討に使う。
    #   MAGS … 2023-04-11上場（当初は Roundhill BIG Tech ETF。2023-11-09に現名称へ変更）。Apple・Microsoft・Alphabet・Amazon・Meta・Nvidia・Teslaの等ウェイト（四半期ごとにリバランス）。
    #          上場から約3.5年と短いため、構成7銘柄の日次データで上場前を再構成して使う（NVDAは取得済み）。
    #          レバレッジ型ではない（一部にトータルリターンスワップを使う）。SBI米国株アプリで売買できるかは未確認。
    "MAGS": "MAGS",
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "META": "META",
    "TSLA": "TSLA",
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


def _download_intraday(ticker, interval, period, prepost=False):
    """分足を取得し、索引をUTCに揃えて返す。prepost=True で時間外（プレ・アフター）の足も含める。"""

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        actions=False,
        prepost=prepost,
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
# (7b) 時間外（プレマーケット・アフターマーケット）の5分足の蓄積（intraday/*_5m_ext.csv）★第43回で追加
#   目的: SBI証券が米国株の時間外取引に対応したとき（2026年内に最長16時間へ拡大予定。日付は未発表）に、
#         「引け後に確定判定を見て、時間外で修正売買する」運用が成り立つかを、実測で確かめるため。
#   測るもの: ①引け後の価格が終値からどれだけ動くか（夜間ギャップのうち、引け後1時間・3時間で何割が起きるか）
#             ②時間外の出来高と足の高低差（スプレッドの代わりの目安。板の気配値そのものは取れない）
#   Yahooの5分足は直近約60日。取り逃すと戻らないので、毎回の取得分を既存ファイルに足す（8週間に1回は実行すること）。
#   対象は SMH と QQQ だけ。*_ext_summary_historical.csv は日ごとの要約（毎回作り直す）。
#   既存の出力・既存の分足ファイルは一切変えない。失敗しても警告を出して続行する。
# ------------------------------------------------------------

EXT_INTRADAY_ENABLED = True

EXT_TICKERS = {
    "SMH": "SMH",
    "QQQ": "QQQ",
}


def _ext_summary(dfx):
    """時間外込みの5分足から、日ごとの要約を作る。
    行の日付 D の「夜」= D のアフター（16:00〜20:00）→ 翌営業日のプレ（4:00〜9:30）→ 翌営業日の寄付き。
    Close1555 = 通常時間の最後の足（15:55開始）の終値 ≒ 終値。
    AH_xxxx = D のアフターで、その時刻までに付いた最後の足の終値。PM_xxxx = 翌営業日のプレで、その時刻までに付いた最後の足の終値。
    Open930_next = 翌営業日の最初の足の始値。Ret_*_bp は Close1555 に対する騰落（bp）。
    Vol_* は出来高、AH_N1700 は引け後1時間に値動きのあった足（高値>安値、または終値≠始値）の本数、*_RangePct は値動きのあった足の高低差/終値（%）の中央値（スプレッドの目安）。
    ★2026-10-15 修正: Yahoo は時間外の5分足の出来高を返さない（全て0）ため、「出来高>0の足だけ使う」と時間外の列が空になる。
      値動きの有無で判断し、値動きのない足は直前の約定価格のまま（足の終値）として扱う。Vol_* は常に0になる（Yahooの制約）。"""

    d = dfx[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    d = d.tz_convert(INTRADAY_TZ).sort_index()
    t = lambda hh, mm: _dt.time(hh, mm)

    byday = {}
    for day, g in d.groupby(d.index.date):
        byday[day] = g.sort_index()

    days = sorted(byday)

    def seg(g, lo, hi):
        tm = g.index.time
        return g[(tm >= lo) & (tm < hi)]

    def last_px(g, lo, hi):
        s = seg(g, lo, hi)
        s = s[s["Close"] > 0]
        return float(s["Close"].iloc[-1]) if len(s) else np.nan

    def moved(s):
        return (s["High"] > s["Low"]) | (s["Close"] != s["Open"])

    def med_range(g, lo, hi):
        s = seg(g, lo, hi)
        s = s[moved(s) & (s["Close"] > 0)]
        return float(((s["High"] - s["Low"]) / s["Close"]).median() * 100) if len(s) else np.nan

    rows = []

    for i, day in enumerate(days):

        g = byday[day]
        rth = seg(g, t(9, 30), t(16, 0))

        if len(rth) == 0:
            continue

        c = float(rth["Close"].iloc[-1])
        nxt = byday[days[i + 1]] if i + 1 < len(days) else None
        nrth = seg(nxt, t(9, 30), t(16, 0)) if nxt is not None else None

        row = {
            "Date": pd.Timestamp(day),
            "LastRegularBar": rth.index[-1].strftime("%H:%M"),    # 15:55 以外なら半日取引か途中の日
            "Close1555": c,
            "AH_1605": last_px(g, t(16, 0), t(16, 5)),
            "AH_1630": last_px(g, t(16, 0), t(16, 30)),
            "AH_1700": last_px(g, t(16, 0), t(17, 0)),
            "AH_1800": last_px(g, t(16, 0), t(18, 0)),
            "AH_2000": last_px(g, t(16, 0), t(20, 0)),
            "AH_N1700": int(moved(seg(g, t(16, 0), t(17, 0))).sum()),
            "Vol_AH1700": float(seg(g, t(16, 0), t(17, 0))["Volume"].sum()),
            "Vol_AH2000": float(seg(g, t(16, 0), t(20, 0))["Volume"].sum()),
            "AH_RangePct1700": med_range(g, t(16, 0), t(17, 0)),
            "RTH_RangePct1500": med_range(g, t(15, 0), t(16, 0)),
            "Vol_RTH": float(rth["Volume"].sum()),
            "PM_0800": np.nan, "PM_0900": np.nan, "PM_0925": np.nan,
            "Vol_PMnext": np.nan, "PM_RangePct0900": np.nan, "Open930_next": np.nan, "NextDate": pd.NaT,
        }

        if nxt is not None and nrth is not None and len(nrth):
            row["PM_0800"] = last_px(nxt, t(4, 0), t(8, 0))
            row["PM_0900"] = last_px(nxt, t(4, 0), t(9, 0))
            row["PM_0925"] = last_px(nxt, t(4, 0), t(9, 25))
            row["Vol_PMnext"] = float(seg(nxt, t(4, 0), t(9, 30))["Volume"].sum())
            row["PM_RangePct0900"] = med_range(nxt, t(8, 0), t(9, 0))
            row["Open930_next"] = float(nrth["Open"].iloc[0])
            row["NextDate"] = pd.Timestamp(days[i + 1])

        for k in ("AH_1605", "AH_1630", "AH_1700", "AH_1800", "AH_2000", "PM_0800", "PM_0900", "PM_0925", "Open930_next"):
            v = row[k]
            row["Ret_" + k + "_bp"] = (v / c - 1.0) * 1e4 if (v == v and c > 0) else np.nan

        rows.append(row)

    return pd.DataFrame(rows).set_index("Date")


if EXT_INTRADAY_ENABLED and INTRADAY_ENABLED:

    print("")
    print("=" * 70)
    print("★追加データ: 時間外の5分足を取得して蓄積中...")
    print("=" * 70)

    try:
        os.makedirs(INTRADAY_DIR, exist_ok=True)
        _ext_ok = True
    except Exception as e:
        _ext_ok = False
        _extra_log("INTRADAY_EXT_DIR", False, f"フォルダ作成失敗: {e!r}")

    if _ext_ok:

        for name, ticker in EXT_TICKERS.items():

            tag = f"INTRA_{name}_5m_ext"
            path = os.path.join(INTRADAY_DIR, f"{name}_5m_ext.csv")

            try:
                new = None

                for attempt in range(2):
                    try:
                        new = _download_intraday(ticker, "5m", "58d", prepost=True)
                    except Exception:
                        new = None
                        if attempt == 1:
                            raise
                    if new is not None:
                        break
                    time.sleep(3)

                if new is None:
                    _extra_log(tag, False, f"{ticker} の時間外込み5分足が空")
                    continue

                merged = _merge_intraday(path, new)

                _tm = merged.tz_convert(INTRADAY_TZ).index.time
                _n_ah = int(sum(1 for x in _tm if _dt.time(16, 0) <= x < _dt.time(20, 0)))
                _n_pm = int(sum(1 for x in _tm if _dt.time(4, 0) <= x < _dt.time(9, 30)))

                _extra_log(
                    tag, True,
                    f"{ticker} 今回{len(new)}本 / 累計{len(merged)}本（アフター{_n_ah}本・プレ{_n_pm}本） "
                    f"{merged.index.min().strftime('%Y-%m-%d')}〜{merged.index.max().strftime('%Y-%m-%d')}"
                )

                if _n_ah == 0:
                    _extra_log(tag + "_AH", False, f"{ticker} にアフターの足が1本も無い（prepost が効いていない可能性）")

                summary = _ext_summary(merged)

                _save_with_stamp(
                    summary,
                    os.path.join(INTRADAY_DIR, f"{name}_ext_summary_historical.csv")
                )

                _extra_log(f"INTRA_{name}_ext_summary", True, f"{len(summary)}日分")

            except Exception as e:
                _extra_log(tag, False, f"{ticker} 取得失敗: {e!r}")


# ============================================================
# ★追加データ（2026-10 追加その3）… GitHub Actions 上で無料で取得・蓄積できるもの
#
#   既存の出力は一切変えない／失敗しても警告を出して続行する、という方針は同じ。
#   目的は、これまで「本環境に無く、検証できなかった」情報を、無料で手元に貯めること。
#
#   (8)  FRB流動性      … FRB総資産・財務省一般勘定(TGA)・翌日物リバースレポ → ネット流動性（週次・2002年末〜）
#   (9)  CFTC建玉       … ナスダック100／S&P500／VIX先物の投資家別ポジション（週次・2006年〜）
#   (10) CBOE put/call  … 2006-11〜2019-10 は静的CSV。2019-10以降は日次JSONを試し、取れた分を蓄積（※URL未確認）
#   (11) FINRA空売り出来高 … 日次（2018-08〜）。市場全体と指定銘柄。公表は取引日の18時（米東部）まで
#   (12) 選択肢の板スナップショット … SMH/QQQ のATM IV・スキュー・put/call比を毎回1行ずつ蓄積（過去分は取得不可）
#   (13) ETF保有銘柄スナップショット … SMH(VanEck)・SOXX(iShares) の保有銘柄ファイルを日付つきで保存（過去分は取得不可）
#   (14) QQQ(ナスダック100)の構成銘柄の履歴 … 公開パッケージ nasdaq-100-ticker-history があれば月次で保存（2007年〜）
#
#   ★使うときの注意（検証時の鉄則）
#     ・週次の系列は「AvailableFrom」列の日付以降にしか使えない（公表遅れ。先読み防止）
#     ・(10)(11)(12)(13) は取得に回数がかかる／過去に遡れないため、毎回の実行で少しずつ溜まる。
#       ワークフローが新しいフォルダ（cftc/ cboe_pc/ finra_shortvol/ options_snapshot/ holdings_snapshot/）を
#       コミット対象に含めているか確認すること（git add -A なら不要）
#     ・URLや形式は取得できない日がある。結果は EXTRA_DATA_STATUS.txt の OK/NG で確認できる
# ============================================================

import io
import json
import hashlib
import re

try:
    import requests
except Exception:      # yfinance の依存で通常は入っている
    requests = None

GH_EXTRA_ENABLED = True            # False にすると、この追加ブロックをまるごと飛ばす
GH_TIME_BUDGET_SEC = 25 * 60       # この追加ブロック全体の時間上限。超えたら重い取得は次回に回す
GH_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; stock-data-ai/1.0; personal research)",
    "Accept": "*/*",
}
_gh_t0 = time.time()


def _gh_time_left():
    return GH_TIME_BUDGET_SEC - (time.time() - _gh_t0)


def _http_get(url, params=None, tries=3, timeout=60):
    """GET。200/403/404 はそのまま返す（呼び出し側で status_code を見る）。
    通信エラーや 429/5xx は少し待って再試行し、だめなら例外にする。"""

    if requests is None:
        raise RuntimeError("requests が使えません")

    last = None

    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=GH_HEADERS, timeout=timeout)
            if r.status_code in (200, 403, 404):
                return r
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last = e
        if i < tries - 1:
            time.sleep(2 * (i + 1))

    raise last if last is not None else RuntimeError("取得失敗")


def _write_dated_csv(df, path, date_col="Date"):
    """date_col の最大日を DataCollectedAt に入れて保存する（索引は保存しない）。"""
    out = df.copy()
    out["DataCollectedAt"] = str(pd.to_datetime(out[date_col]).max())[:10]
    out.to_csv(path, index=False)
    return out


def _accumulate_csv(path, new, key_cols):
    """既存CSVに new を足す。key_cols が同じ行は今回分を優先。既存ファイルが読めないときは例外（上書きしない）。"""

    new = new.copy()

    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old.drop(columns=[c for c in ["DataCollectedAt"] if c in old.columns])
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new

    merged = merged.drop_duplicates(subset=key_cols, keep="last")
    merged = merged.sort_values(key_cols).reset_index(drop=True)

    return merged


def _read_day_list(path):
    """「データなし」と確認した日付の一覧（休場日の再取得を避けるため）。"""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {x.strip() for x in f if x.strip()}


def _append_day_list(path, days):
    if not days:
        return
    cur = _read_day_list(path)
    cur |= {pd.Timestamp(d).strftime("%Y-%m-%d") for d in days}
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(cur)) + "\n")


def _only_bracketed(days, known):
    """「データなし」と記録してよい日だけ残す。取得済みの日付の最古と最新の間にある日（休場日）に限る。
    取れた日が1つもない、または範囲の外の日は、通信やブロックの失敗かもしれないので記録しない。"""
    ks = sorted(pd.Timestamp(k) for k in known)
    if not ks:
        return []
    lo, hi = ks[0], ks[-1]
    return [d for d in days if lo < d < hi]


def _fred_series(sid):
    """FREDの1系列を Date 索引・Value 列で返す（既存のFRED取得と同じ方法）。"""
    df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
    if "observation_date" not in df.columns or sid not in df.columns:
        raise RuntimeError(f"{sid}: 列が想定と違います {list(df.columns)}")
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df = df.rename(columns={"observation_date": "Date", sid: "Value"})[["Date", "Value"]]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    return df.dropna(subset=["Value"]).sort_values("Date").set_index("Date")


# ------------------------------------------------------------
# (8) FRB流動性（FRED）と、財務省の日次TGA
#   WALCL    … FRB総資産（水曜時点・百万ドル）。H.4.1は木曜16:30(米東部)公表
#   WDTGAL   … 財務省一般勘定(TGA)（水曜時点・百万ドル）
#   RRPONTSYD… 翌日物リバースレポ残高（日次・十億ドル）
#   ネット流動性 = WALCL − WDTGAL − RRPONTSYD（十億ドル）… 市場に出回る資金量の代理指標としてよく使われる
#   ※ AvailableFrom 列より前の日には使えない（先読み防止）
# ------------------------------------------------------------

LIQ_FRED = {
    # 保存名: (FRED ID, 公表遅れ（日）, 単位)
    "FedAssets": ("WALCL", 2, "百万ドル"),
    "TGA_Wed": ("WDTGAL", 2, "百万ドル"),
    "ON_RRP": ("RRPONTSYD", 1, "十億ドル"),
    "BreakEven10Y": ("T10YIE", 1, "%"),
    "SOFR": ("SOFR", 1, "%"),
}


def _fetch_dts_tga(start="2005-01-01"):
    """財務省 Fiscal Data の日次財政報告(DTS)から、TGAの終値残高（百万ドル）を取る。
    2022年以降の表記 'Treasury General Account (TGA) Closing Balance' を優先し、
    それ以前は 'Federal Reserve Account' で補う（※旧表記の行は未確認）。"""

    base = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/"
            "accounting/dts/operating_cash_balance")

    rows = []
    page = 1

    while page <= 30:
        r = _http_get(base, params={
            "fields": "record_date,account_type,open_today_bal,close_today_bal",
            "filter": f"record_date:gte:{start},table_nbr:eq:I",
            "sort": "record_date",
            "page[size]": 10000,
            "page[number]": page,
        })
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        j = r.json()
        data = j.get("data", [])
        rows += data
        total_pages = int(j.get("meta", {}).get("total-pages", page))
        if page >= total_pages or not data:
            break
        page += 1

    if not rows:
        raise RuntimeError("有効な行がありません")

    df = pd.DataFrame(rows)
    df = df[df["account_type"].isin(["Treasury General Account (TGA) Closing Balance", "Federal Reserve Account"])].copy()
    if df.empty:
        raise RuntimeError("TGAの行が見つかりません（表記が変わった可能性）")
    df["Date"] = pd.to_datetime(df["record_date"])
    close = pd.to_numeric(df["close_today_bal"], errors="coerce")
    open_ = pd.to_numeric(df["open_today_bal"], errors="coerce")
    df["Value"] = close.where(close.notna(), open_)
    df["Pref"] = np.where(df["account_type"].str.contains("TGA", na=False), 0, 1)
    df = df.dropna(subset=["Value"]).sort_values(["Date", "Pref"])
    df = df.drop_duplicates(subset=["Date"], keep="first")

    return df[["Date", "Value"]].set_index("Date").sort_index()


def _build_net_liquidity(fed_assets, tga_wed, on_rrp):
    """水曜時点のネット流動性（十億ドル）を作る。入力は Date 索引・Value 列。"""

    nl = pd.DataFrame(index=fed_assets.index)
    nl.index.name = "Date"
    nl["FedAssets_bn"] = fed_assets["Value"] / 1000.0
    nl["TGA_bn"] = tga_wed["Value"].reindex(nl.index) / 1000.0
    rrp = on_rrp["Value"].reindex(nl.index, method="ffill", tolerance=pd.Timedelta(days=4))
    first_rrp = on_rrp.index.min()
    # RRPの系列が始まる前の日は 0 とみなす（始まった後に取れない日は空欄のまま）
    nl["ON_RRP_bn"] = rrp.where(~((nl.index < first_rrp) & rrp.isna()), 0.0)
    nl["NetLiquidity_bn"] = nl["FedAssets_bn"] - nl["TGA_bn"] - nl["ON_RRP_bn"]
    nl["NetLiq_Chg4w_bn"] = nl["NetLiquidity_bn"].diff(4)
    nl["NetLiq_Chg13w_bn"] = nl["NetLiquidity_bn"].diff(13)
    nl["AvailableFrom"] = nl.index + pd.Timedelta(days=2)
    return nl.dropna(subset=["NetLiquidity_bn"])


def run_block_liquidity():

    got = {}

    for name, (sid, lag, unit) in LIQ_FRED.items():

        try:
            df = _fred_series(sid)
            df["AvailableFrom"] = df.index + pd.Timedelta(days=lag)
            _save_with_stamp(df, f"{name}_historical.csv")
            got[name] = df
            _extra_log(f"LIQ_{name}", True,
                       f"{sid}（{unit}） {df.index.min().date()}〜{df.index.max().date()} {len(df)}件")
        except Exception as e:
            _extra_log(f"LIQ_{name}", False, f"{sid} 取得失敗: {e!r}")

    try:
        for k in ("FedAssets", "TGA_Wed", "ON_RRP"):
            if k not in got:
                raise RuntimeError(f"{k} が取れていないため計算できません")
        nl = _build_net_liquidity(got["FedAssets"], got["TGA_Wed"], got["ON_RRP"])
        _save_with_stamp(nl, "NET_LIQUIDITY_historical.csv")
        _extra_log("NET_LIQUIDITY", True,
                   f"{nl.index.min().date()}〜{nl.index.max().date()} {len(nl)}件"
                   f"（最新 {nl['NetLiquidity_bn'].iloc[-1]:.0f}十億ドル）")
    except Exception as e:
        _extra_log("NET_LIQUIDITY", False, f"計算失敗: {e!r}")

    try:
        tga = _fetch_dts_tga()
        tga["AvailableFrom"] = tga.index + pd.Timedelta(days=2)
        _save_with_stamp(tga, "TGA_Daily_historical.csv")
        _extra_log("TGA_Daily", True,
                   f"財務省DTS（百万ドル） {tga.index.min().date()}〜{tga.index.max().date()} {len(tga)}件")
    except Exception as e:
        _extra_log("TGA_Daily", False, f"取得失敗: {e!r}")


# ------------------------------------------------------------
# (9) CFTC 建玉明細（Traders in Financial Futures・先物のみ）
#   公式の公開API（認証不要）。レポート日は火曜、公表は金曜15:30(米東部)。
#   → 先読み防止のため AvailableFrom = レポート日 + 4日（次の月曜から使う）
#   Dealer=ディーラー / AssetMgr=資産運用会社 / LevMoney=レバレッジファンド（ヘッジファンド等）
#   対象は、ナスダック100・S&P500・VIX先物（名前の一部で絞る）。見つかった市場名は結果一覧に残す
# ------------------------------------------------------------

CFTC_TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.json"
CFTC_WHERE = ("market_and_exchange_names like '%NASDAQ-100%' OR "
              "market_and_exchange_names like '%S&P 500%' OR "
              "market_and_exchange_names like '%VIX FUTURES%'")
CFTC_FIELDS = [
    "report_date_as_yyyy_mm_dd", "market_and_exchange_names", "cftc_contract_market_code",
    "open_interest_all",
    "dealer_positions_long_all", "dealer_positions_short_all",
    "asset_mgr_positions_long", "asset_mgr_positions_short",
    "lev_money_positions_long", "lev_money_positions_short",
    "other_rept_positions_long", "other_rept_positions_short",
    "nonrept_positions_long_all", "nonrept_positions_short_all",
    "traders_tot_all",
]
CFTC_SLUGS = [
    (r"^NASDAQ-100 CONSOLIDATED", "NQ_CONS"),
    (r"^MICRO E-MINI NASDAQ", "MNQ"),
    (r"NASDAQ-100", "NQ_MINI"),
    (r"^S&P 500 CONSOLIDATED", "ES_CONS"),
    (r"^MICRO E-MINI S&P", "MES"),
    (r"^E-MINI S&P 500", "ES_EMINI"),
    (r"^VIX FUTURES", "VX"),
]


def _cftc_slug(name):
    up = str(name).upper()
    for pat, slug in CFTC_SLUGS:
        if re.search(pat, up):
            return slug
    return re.sub(r"[^A-Z0-9]+", "_", up)[:30].strip("_") or "OTHER"


def _cftc_prepare(rows):
    """APIの行（文字列の数値）を、日付・ネット建玉つきの表にする。"""

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"].astype(str).str[:10])
    df["Market"] = df["market_and_exchange_names"].astype(str)

    num = [c for c in CFTC_FIELDS if c not in
           ("report_date_as_yyyy_mm_dd", "market_and_exchange_names", "cftc_contract_market_code")]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.rename(columns={
        "open_interest_all": "OpenInterest",
        "dealer_positions_long_all": "Dealer_Long", "dealer_positions_short_all": "Dealer_Short",
        "asset_mgr_positions_long": "AssetMgr_Long", "asset_mgr_positions_short": "AssetMgr_Short",
        "lev_money_positions_long": "LevMoney_Long", "lev_money_positions_short": "LevMoney_Short",
        "other_rept_positions_long": "OtherRept_Long", "other_rept_positions_short": "OtherRept_Short",
        "nonrept_positions_long_all": "NonRept_Long", "nonrept_positions_short_all": "NonRept_Short",
        "traders_tot_all": "TradersTotal",
        "cftc_contract_market_code": "ContractCode",
    })

    for g in ("Dealer", "AssetMgr", "LevMoney", "OtherRept", "NonRept"):
        df[f"{g}_Net"] = df[f"{g}_Long"] - df[f"{g}_Short"]

    oi = df["OpenInterest"].replace(0, np.nan)
    df["LevMoney_Net_pctOI"] = df["LevMoney_Net"] / oi * 100
    df["AssetMgr_Net_pctOI"] = df["AssetMgr_Net"] / oi * 100
    df["Dealer_Net_pctOI"] = df["Dealer_Net"] / oi * 100
    df["AvailableFrom"] = df["Date"] + pd.Timedelta(days=4)
    df["Slug"] = df["Market"].map(_cftc_slug)

    keep = ["Date", "AvailableFrom", "Market", "Slug", "ContractCode", "OpenInterest",
            "Dealer_Long", "Dealer_Short", "Dealer_Net", "Dealer_Net_pctOI",
            "AssetMgr_Long", "AssetMgr_Short", "AssetMgr_Net", "AssetMgr_Net_pctOI",
            "LevMoney_Long", "LevMoney_Short", "LevMoney_Net", "LevMoney_Net_pctOI",
            "OtherRept_Long", "OtherRept_Short", "OtherRept_Net",
            "NonRept_Long", "NonRept_Short", "NonRept_Net", "TradersTotal"]

    return df[keep].sort_values(["Slug", "Market", "Date"]).reset_index(drop=True)


def run_block_cftc():

    rows = []
    offset = 0
    limit = 50000

    while True:
        r = _http_get(CFTC_TFF_URL, params={
            "$select": ",".join(CFTC_FIELDS),
            "$where": CFTC_WHERE,
            "$order": "report_date_as_yyyy_mm_dd,id",
            "$limit": limit,
            "$offset": offset,
        })
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        chunk = r.json()
        rows += chunk
        if len(chunk) < limit:
            break
        offset += limit

    if not rows:
        raise RuntimeError("有効な行がありません")

    df = _cftc_prepare(rows)

    os.makedirs("cftc", exist_ok=True)

    for slug, g in df.groupby("Slug"):
        g = g.drop(columns=["Slug"])
        _write_dated_csv(g, os.path.join("cftc", f"{slug}_TFF_historical.csv"))
        names = "／".join(sorted(g["Market"].unique()))[:120]
        _extra_log(f"CFTC_{slug}", True,
                   f"{g['Date'].min().date()}〜{g['Date'].max().date()} {len(g)}行 [{names}]")


# ------------------------------------------------------------
# (10) CBOE の put/call 比
#   静的CSV（2006-11-01〜2019-10-04。以降は更新されない）
#   2019-10 以降は日次の統計ページが使うJSONを試す（※URLと形式は未確認。取れた日だけ蓄積する）
# ------------------------------------------------------------

CBOE_PC_BASE = "https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/"
CBOE_PC_STATIC = {
    "TOTAL": "totalpc.csv",
    "INDEX": "indexpc.csv",
    "EQUITY": "equitypc.csv",
    "ETP": "etppc.csv",
    "VIX": "vixpc.csv",
}
CBOE_PC_DAILY_URL = "https://cdn.cboe.com/data/us/options/market_statistics/daily/{d}_daily_options"
CBOE_PC_DAILY_START = "2019-10-07"      # 静的CSVの最終日の翌営業日
CBOE_PC_MAX_PER_RUN = 150


def _parse_cboe_pc_csv(text):
    """「DATE,CALLS,PUTS,TOTAL,P/C Ratio」の見出し行を探して読む（前置きの注記行は読み飛ばす）。"""

    lines = text.splitlines()
    k = next((i for i, l in enumerate(lines) if l.strip().upper().startswith("DATE,")), None)

    if k is None:
        raise RuntimeError("見出し行（DATE,…）が見つかりません")

    df = pd.read_csv(io.StringIO("\n".join(lines[k:])))
    df.columns = [str(c).strip().upper() for c in df.columns]

    d = pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce")
    if d.isna().all():
        d = pd.to_datetime(df["DATE"], errors="coerce")
    df["DATE"] = d

    df = df.dropna(subset=["DATE"])

    for c in df.columns:
        if c != "DATE":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.rename(columns={"P/C RATIO": "PCRATIO"})

    return df.sort_values("DATE").drop_duplicates("DATE", keep="last").set_index("DATE")


def _cboe_pc_daily_json(day):
    """1日分の JSON から「…PUT/CALL RATIO」という名前の値を全部拾う。取れなければ None。"""

    r = _http_get(CBOE_PC_DAILY_URL.format(d=day.strftime("%Y-%m-%d")), tries=2, timeout=30)

    if r.status_code != 200:
        return None

    try:
        j = r.json()
    except Exception:
        return None

    found = {}

    def walk(o):
        if isinstance(o, dict):
            nm = o.get("name")
            val = o.get("value")
            if isinstance(nm, str) and "PUT/CALL" in nm.upper() and val is not None:
                try:
                    found[nm.strip().upper()] = float(str(val).replace(",", ""))
                except Exception:
                    pass
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(j)

    return found or None


def run_block_cboe_pc():

    os.makedirs("cboe_pc", exist_ok=True)

    # (a) 静的CSV
    for name, fn in CBOE_PC_STATIC.items():
        try:
            r = _http_get(CBOE_PC_BASE + fn)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            df = _parse_cboe_pc_csv(r.text)
            _save_with_stamp(df, os.path.join("cboe_pc", f"PC_{name}_historical.csv"))
            _extra_log(f"CBOE_PC_{name}", True,
                       f"{df.index.min().date()}〜{df.index.max().date()} {len(df)}件（静的CSV）")
        except Exception as e:
            _extra_log(f"CBOE_PC_{name}", False, f"{fn} 取得失敗: {e!r}")

    # (b) 2019-10以降の日次JSON（蓄積）
    path = os.path.join("cboe_pc", "CBOE_PC_daily_historical.csv")
    nodata_path = os.path.join("cboe_pc", "_nodata_days.txt")

    try:
        have = set()
        if os.path.exists(path):
            have = set(pd.read_csv(path)["Date"].astype(str).str[:10])

        nodata = _read_day_list(nodata_path)

        today = pd.Timestamp(_dt.datetime.now(_dt.timezone.utc).date())
        days = [d for d in pd.bdate_range(CBOE_PC_DAILY_START, today)
                if d.strftime("%Y-%m-%d") not in have and d.strftime("%Y-%m-%d") not in nodata]
        days = sorted(days, reverse=True)[:CBOE_PC_MAX_PER_RUN]

        new_rows = []
        no_new = []
        n_try = 0

        for d in days:
            if _gh_time_left() < 300:
                break
            try:
                got = _cboe_pc_daily_json(d)
            except Exception:
                got = "ERR"        # 通信エラー等。「データなし」には記録しない
            n_try += 1
            # 最初の8日が全滅なら、URLか形式が違う（または通信できない）とみなして打ち切る
            if n_try >= 8 and not new_rows and (got is None or isinstance(got, str)):
                break
            if isinstance(got, str):
                continue
            if got is None:
                # 直近3営業日は未公表かもしれないので「データなし」に記録しない
                if (today - d).days > 5:
                    no_new.append(d)
            else:
                row = {"Date": d.strftime("%Y-%m-%d")}
                row.update(got)
                new_rows.append(row)
            time.sleep(0.2)

        if new_rows:
            new = pd.DataFrame(new_rows)
            merged = _accumulate_csv(path, new, ["Date"])
            _write_dated_csv(merged, path)
            _append_day_list(nodata_path, _only_bracketed(no_new, have | set(new["Date"])))
            _extra_log("CBOE_PC_DAILY", True,
                       f"今回{len(new_rows)}日 / 累計{len(merged)}日 "
                       f"{merged['Date'].min()}〜{merged['Date'].max()}")
        elif have:
            _append_day_list(nodata_path, _only_bracketed(no_new, have))
            _extra_log("CBOE_PC_DAILY", True,
                       f"今回の新規なし（累計{len(have)}日、最新 {max(have)}。{n_try}日試して公表なし）")
        else:
            _extra_log("CBOE_PC_DAILY", False,
                       f"{n_try}日試して1日も取れず（URLまたは形式が違う可能性。静的CSVは別途OK）")

    except Exception as e:
        _extra_log("CBOE_PC_DAILY", False, f"失敗: {e!r}")


# ------------------------------------------------------------
# (11) FINRA 日次の空売り出来高（取引所外 TRF/ADF の合算 CNMS ファイル）
#   2018-08-01 から。取引日の18時(米東部)までに公表。区切りは「|」
#   ・市場全体の空売り比率 … FINRA_SHORTVOL_market_historical.csv
#   ・指定銘柄の空売り比率 … FINRA_SHORTVOL_symbols_historical.csv
#   1回の実行で FINRA_MAX_PER_RUN 日分まで。直近→過去の順に埋める（全期間は十数回で埋まる）
# ------------------------------------------------------------

FINRA_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{d}.txt"
FINRA_START = "2018-08-01"
FINRA_MAX_PER_RUN = 120
FINRA_SYMBOLS = ["SMH", "QQQ", "SOXX", "SOXL", "SOXS", "TLT", "IEF", "SPY"] + SEMI_CONSTITUENTS


def _parse_finra_day(text, day):
    """1日分のファイルを読む。戻り値: (指定銘柄の表, 市場全体の1行の辞書)。末尾の件数行などは捨てる。"""

    df = pd.read_csv(io.StringIO(text), sep="|", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]

    for c in ("Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume"):
        if c not in df.columns:
            raise RuntimeError(f"列 {c} がありません: {list(df.columns)}")

    df = df[df["Date"].astype(str).str.fullmatch(r"\d{8}", na=False)].copy()

    for c in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["TotalVolume"])

    if df.empty:
        raise RuntimeError("有効な行がありません")

    tot = float(df["TotalVolume"].sum())
    shr = float(df["ShortVolume"].sum())

    market = {
        "Date": day.strftime("%Y-%m-%d"),
        "ShortVolume": shr,
        "TotalVolume": tot,
        "ShortRatioPct": (shr / tot * 100) if tot > 0 else np.nan,
        "NSymbols": int(len(df)),
    }

    sel = df[df["Symbol"].isin(FINRA_SYMBOLS)][
        ["Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume"]
    ].copy()
    sel.insert(0, "Date", day.strftime("%Y-%m-%d"))
    sel["ShortRatioPct"] = sel["ShortVolume"] / sel["TotalVolume"].replace(0, np.nan) * 100

    return sel, market


def run_block_finra():

    os.makedirs("finra_shortvol", exist_ok=True)

    mpath = os.path.join("finra_shortvol", "FINRA_SHORTVOL_market_historical.csv")
    spath = os.path.join("finra_shortvol", "FINRA_SHORTVOL_symbols_historical.csv")
    nodata_path = os.path.join("finra_shortvol", "_nodata_days.txt")

    have = set()
    if os.path.exists(mpath):
        have = set(pd.read_csv(mpath)["Date"].astype(str).str[:10])

    nodata = _read_day_list(nodata_path)

    today = pd.Timestamp(_dt.datetime.now(_dt.timezone.utc).date())
    days = [d for d in pd.bdate_range(FINRA_START, today)
            if d.strftime("%Y-%m-%d") not in have and d.strftime("%Y-%m-%d") not in nodata]
    days = sorted(days, reverse=True)[:FINRA_MAX_PER_RUN]

    mrows, srows, no_new = [], [], []
    n_err = 0

    for d in days:

        if _gh_time_left() < 300:
            break

        try:
            r = _http_get(FINRA_URL.format(d=d.strftime("%Y%m%d")), tries=2, timeout=90)

            if r.status_code != 200:
                # 直近3営業日は未公表かもしれないので「データなし」に記録しない
                if (today - d).days > 5:
                    no_new.append(d)
                continue

            sel, market = _parse_finra_day(r.text, d)
            mrows.append(market)
            srows.append(sel)

        except Exception:
            n_err += 1
            if n_err >= 5 and not mrows:
                break      # 最初から失敗が続くなら打ち切る

        time.sleep(0.2)

    if mrows:
        m_new = pd.DataFrame(mrows)
        s_new = pd.concat(srows, ignore_index=True)
        m_all = _accumulate_csv(mpath, m_new, ["Date"])
        s_all = _accumulate_csv(spath, s_new, ["Date", "Symbol"])
        _write_dated_csv(m_all, mpath)
        _write_dated_csv(s_all, spath)
        _append_day_list(nodata_path, _only_bracketed(no_new, have | set(m_new["Date"])))
        _extra_log("FINRA_SHORTVOL", True,
                   f"今回{len(mrows)}日 / 累計{len(m_all)}日 {m_all['Date'].min()}〜{m_all['Date'].max()}"
                   f"（銘柄 {s_all['Symbol'].nunique()}）")
    elif have and n_err == 0:
        _append_day_list(nodata_path, _only_bracketed(no_new, have))
        _extra_log("FINRA_SHORTVOL", True,
                   f"今回の新規なし（累計{len(have)}日、最新 {max(have)}。{len(days)}日試して公表なし）")
    else:
        _extra_log("FINRA_SHORTVOL", False,
                   f"取れた日なし（失敗 {n_err}件／対象 {len(days)}日。URLまたは通信を確認）")


# ------------------------------------------------------------
# (12) SMH / QQQ の選択肢（オプション）の板スナップショット
#   SMH固有のIV（VIXのSMH版）は過去分を取れないので、毎回1行ずつ蓄積する。
#   ・IV30_ATM … 約30日先のATMのインプライド・ボラティリティ（年率、小数。0.35 = 35%）
#   ・Skew90   … 行使価格が現値の90%のプットIV − ATM IV（暴落への警戒度）
#   ・PC_Volume／PC_OI … 3〜75日先の満期の出来高／建玉の プット÷コール
#   ※ Yahooの板は取引時間外にIVが乱れることがある。NValidATM（有効なATMの満期数）も残す
# ------------------------------------------------------------

OPT_SYMBOLS = ["SMH", "QQQ"]
OPT_DTE_MIN, OPT_DTE_MAX = 3, 75


def _nearest_iv(df, target, tol=0.03, spot=None):
    """行使価格が target に最も近い行のIV。現値比で tol を超えて離れていれば NaN。"""
    if df is None or len(df) == 0:
        return np.nan
    d = df.copy()
    d["strike"] = pd.to_numeric(d["strike"], errors="coerce")
    d["iv"] = pd.to_numeric(d["impliedVolatility"], errors="coerce")
    d = d[(d["iv"] > 0.05) & d["strike"].notna()]
    if d.empty:
        return np.nan
    i = (d["strike"] - target).abs().idxmin()
    if spot and abs(d.loc[i, "strike"] - target) / spot > tol:
        return np.nan
    return float(d.loc[i, "iv"])


def _opt_expiry_stats(calls, puts, spot):
    """1満期分: ATM IV、プット90%IV、コール110%IV、出来高・建玉。"""

    c_atm = _nearest_iv(calls, spot, spot=spot)
    p_atm = _nearest_iv(puts, spot, spot=spot)
    atm = np.nanmean([c_atm, p_atm]) if not (np.isnan(c_atm) and np.isnan(p_atm)) else np.nan

    def s(df, col):
        return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()) if df is not None and len(df) else 0.0

    return {
        "atm": atm,
        "put90": _nearest_iv(puts, spot * 0.9, spot=spot),
        "call110": _nearest_iv(calls, spot * 1.1, spot=spot),
        "cvol": s(calls, "volume"), "pvol": s(puts, "volume"),
        "coi": s(calls, "openInterest"), "poi": s(puts, "openInterest"),
    }


def _interp_30d(items, key, target=30.0):
    """(満期までの日数, 統計) の列から、約30日先の値を総分散で内挿する。"""

    pts = [(dte, st[key]) for dte, st in items if not np.isnan(st[key])]
    if not pts:
        return np.nan, "none"
    pts.sort()
    lo = [p for p in pts if p[0] <= target]
    hi = [p for p in pts if p[0] > target]
    if lo and hi:
        d1, v1 = lo[-1]
        d2, v2 = hi[0]
        var1, var2 = v1 * v1 * d1 / 365.0, v2 * v2 * d2 / 365.0
        w = (target - d1) / (d2 - d1)
        var = var1 + w * (var2 - var1)
        return float(np.sqrt(var * 365.0 / target)), "interp"
    nearest = min(pts, key=lambda p: abs(p[0] - target))
    return float(nearest[1]), "nearest"


def _option_snapshot(symbol, asof, spot):

    t = yf.Ticker(symbol)
    exps = list(t.options or [])

    items = []

    for e in exps:
        dte = (pd.Timestamp(e) - asof).days
        if dte < OPT_DTE_MIN or dte > OPT_DTE_MAX:
            continue
        try:
            ch = t.option_chain(e)
            items.append((dte, _opt_expiry_stats(ch.calls, ch.puts, spot)))
        except Exception:
            continue      # この満期だけ飛ばす

    if not items:
        raise RuntimeError("対象の満期がありません")

    iv30, how = _interp_30d(items, "atm")
    p90, _ = _interp_30d(items, "put90")
    c110, _ = _interp_30d(items, "call110")

    cv = sum(st["cvol"] for _, st in items)
    pv = sum(st["pvol"] for _, st in items)
    co = sum(st["coi"] for _, st in items)
    po = sum(st["poi"] for _, st in items)

    near_dte, near_st = min(items, key=lambda x: x[0])

    return {
        "Date": asof.strftime("%Y-%m-%d"),
        "SnapshotUTC": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "Spot": spot,
        "IV30_ATM": iv30,
        "IV30_Put90": p90,
        "IV30_Call110": c110,
        "Skew90": (p90 - iv30) if not (np.isnan(p90) or np.isnan(iv30)) else np.nan,
        "IV_Near": near_st["atm"],
        "DTE_Near": near_dte,
        "PC_Volume": (pv / cv) if cv > 0 else np.nan,
        "PC_OI": (po / co) if co > 0 else np.nan,
        "CallVol": cv, "PutVol": pv, "CallOI": co, "PutOI": po,
        "NExp": len(items),
        "NValidATM": int(sum(1 for _, st in items if not np.isnan(st["atm"]))),
        "IV30_Method": how,
    }


def run_block_options():

    os.makedirs("options_snapshot", exist_ok=True)

    for sym in OPT_SYMBOLS:

        try:
            if sym not in all_data:
                raise RuntimeError(f"{sym} の日足が無いため基準日と現値を決められません")

            asof = pd.Timestamp(all_data[sym].index.max())
            spot = float(all_data[sym]["Close"].iloc[-1])

            row = _option_snapshot(sym, asof, spot)

            path = os.path.join("options_snapshot", f"{sym}_options_snapshot.csv")
            merged = _accumulate_csv(path, pd.DataFrame([row]), ["Date"])
            _write_dated_csv(merged, path)

            iv = row["IV30_ATM"]
            _extra_log(f"OPT_{sym}", True,
                       f"{row['Date']} IV30={iv:.3f} Skew90={row['Skew90']:.3f} PC_Vol={row['PC_Volume']:.2f} "
                       f"満期{row['NExp']}本（累計{len(merged)}日）"
                       if not np.isnan(iv) else
                       f"{row['Date']} IVが取れず（満期{row['NExp']}本・累計{len(merged)}日）")

        except Exception as e:
            _extra_log(f"OPT_{sym}", False, f"失敗: {e!r}")


# ------------------------------------------------------------
# (13) ETF保有銘柄のスナップショット
#   SMHの時点ごとの保有銘柄は、公式にも過去分が公開されていない。今日から毎回保存して溜める。
#   ・SMH … VanEck の保有銘柄ファイル(xlsx)を原本のまま日付つきで保存（取れなければ yfinance の上位10銘柄）
#   ・SOXX… iShares の保有銘柄ファイル(csv)を原本のまま保存
#   原本を保存するので、表の形式が変わっても後から読み直せる。内容が前回と同じなら保存しない
# ------------------------------------------------------------

HOLDINGS_SOURCES = [
    # (名前, URL, 拡張子, 中身の確認関数)
    ("SMH", "https://www.vaneck.com/us/en/etf/equity/smh/holdings/download/xlsx/", "xlsx",
     lambda b: b[:2] == b"PK"),
    ("SOXX",
     "https://www.ishares.com/us/products/239705/ishares-phlx-semiconductor-etf/"
     "1467271812596.ajax?fileType=csv&fileName=SOXX_holdings&dataType=fund", "csv",
     lambda b: b"Ticker" in b[:5000]),
]


def _save_raw_snapshot(name, url, ext, check, today):
    """原本を holdings_snapshot/<名前>/<名前>_holdings_<日付>.<拡張子> に保存。戻り値: 'saved' / 'same'。"""

    r = _http_get(url, tries=2, timeout=90)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")

    body = r.content

    if not check(body):
        raise RuntimeError("中身が想定と違います（ボット対策の画面などの可能性）")

    folder = os.path.join("holdings_snapshot", name)
    os.makedirs(folder, exist_ok=True)

    h = hashlib.md5(body).hexdigest()
    hpath = os.path.join(folder, "_last_md5.txt")

    if os.path.exists(hpath) and open(hpath, "r", encoding="utf-8").read().strip() == h:
        return "same"

    with open(os.path.join(folder, f"{name}_holdings_{today}.{ext}"), "wb") as f:
        f.write(body)

    with open(hpath, "w", encoding="utf-8") as f:
        f.write(h)

    return "saved"


def run_block_holdings():

    today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")

    smh_ok = False

    for name, url, ext, check in HOLDINGS_SOURCES:
        try:
            res = _save_raw_snapshot(name, url, ext, check, today)
            _extra_log(f"HOLD_{name}", True, f"{today} 原本を保存（{'前回と同じ内容のため省略' if res == 'same' else '新規'}）")
            if name == "SMH":
                smh_ok = True
        except Exception as e:
            _extra_log(f"HOLD_{name}", False, f"保有銘柄ファイルを取得できず: {e!r}")

    # SMH の原本が取れなかった日の代替: yfinance の上位保有銘柄（上位10程度。比率つき）
    if not smh_ok:
        try:
            th = yf.Ticker("SMH").funds_data.top_holdings
            if th is None or len(th) == 0:
                raise RuntimeError("上位保有銘柄が空")
            df = th.reset_index()
            df.columns = [str(c) for c in df.columns]
            df.insert(0, "Date", today)
            os.makedirs(os.path.join("holdings_snapshot", "SMH"), exist_ok=True)
            path = os.path.join("holdings_snapshot", "SMH", "SMH_top_holdings_yf.csv")
            key = [df.columns[1]]
            merged = _accumulate_csv(path, df, ["Date"] + key)
            _write_dated_csv(merged, path)
            _extra_log("HOLD_SMH_YF", True, f"{today} 上位{len(df)}銘柄（累計{merged['Date'].nunique()}日）")
        except Exception as e:
            _extra_log("HOLD_SMH_YF", False, f"代替取得も失敗: {e!r}")


# ------------------------------------------------------------
# (14) QQQ(ナスダック100)の構成銘柄の履歴（月初時点）
#   公開パッケージ nasdaq-100-ticker-history（MIT、2007-02〜）を使う。入っていなければ飛ばす。
#   ワークフローに  pip install nasdaq-100-ticker-history  を足すと有効になる。
#   用途: 「今の構成銘柄で過去を見る」生存者バイアスの大きさを測る。
# ------------------------------------------------------------

def run_block_qqq_members():

    try:
        from nasdaq_100_ticker_history import tickers_as_of
    except Exception:
        _extra_log("QQQ_MEMBERS_PIT", False,
                   "nasdaq-100-ticker-history が未導入（pip install nasdaq-100-ticker-history を追加すると有効）")
        return

    rows = []
    d = pd.Timestamp("2007-02-01")
    end = pd.Timestamp(_dt.datetime.now(_dt.timezone.utc).date())

    while d <= end:
        try:
            tk = sorted(tickers_as_of(d.year, d.month, 1))
            rows.append({"Date": d.strftime("%Y-%m-%d"), "N": len(tk), "Tickers": " ".join(tk)})
        except Exception:
            pass
        d = (d + pd.offsets.MonthBegin(1))

    if not rows:
        raise RuntimeError("1か月分も取れませんでした")

    df = pd.DataFrame(rows)
    _write_dated_csv(df, "QQQ_MEMBERS_PIT_historical.csv")
    _extra_log("QQQ_MEMBERS_PIT", True,
               f"{df['Date'].min()}〜{df['Date'].max()} {len(df)}か月分（最新 {int(df['N'].iloc[-1])}銘柄）")


# ============================================================
# ★追加データ（2026-10 追加その4）… 決算・産業統計まわりの無料データ（連絡先もキーも要らない・通信が少ないものだけ）
#
#   背景: 第86回の検証で、決算の数値・産業統計が『検証する環境から取れず』未検証のまま残った。
#         GitHub Actions は通信が開いているので、ここで毎回貯める。
#   方針は同じ: 既存の出力は一切変えない／各ブロックは独立／失敗しても続行／結果は EXTRA_DATA_STATUS.txt。
#   ★Secrets の登録は要らない。名前・連絡先・APIキーが要るもの（SEC EDGAR・韓国DART・FRED の改定履歴API）は入れていない。
#
#   (15) 産業統計(FRED)   … 半導体の生産・稼働率、電子製品の出荷・在庫・受注・受注残、半導体の生産者物価（月次）
#                          industry_stats/  ＋ 『初めて見えた日』のログ（後の改定で当時の値が消えない）
#   (16) 台湾の月次売上    … TSMC・UMC・MediaTek・Nanya・AIサーバー系（TWSE オープンAPI）。毎月10日ごろ公表
#                          taiwan_revenue/  TSMC は取得済みの2012年〜の履歴を種として同梱
#
#   ★使うときの注意（検証時の鉄則）
#     ・先読みしない。FRED・台湾の AvailableFrom は規則で決めた『推定』（保守的）。初めて見えた日（SeenUTC / FirstSeen / AsOf）が
#       分かるものは、そちらが正確。Backfill=1 / Src=backfill / Src=trend_backfill は『実際に見えた日ではない』初回の一括分。
#     ・FRED の通常の取得は『改定後の値』。当時の値は industry_stats/_observed_values.csv（取り始めてから）に残る。
#     ・この追加は『実際に接続して確かめたもの』ではない（作った環境からは外部サイトに届かない）。
#       最初の実行後に EXTRA_DATA_STATUS.txt の OK/NG を確認し、NG の理由（HTTP の番号と応答の先頭）を見て直す。
# ============================================================

X4_TIME_BUDGET_SEC = 20 * 60       # 追加その4 の専用の時間上限（既存ブロックの予算とは別）
_x4_t0 = None


def _x4_left():
    global _x4_t0
    if _x4_t0 is None:
        _x4_t0 = time.time()
    return X4_TIME_BUDGET_SEC - (time.time() - _x4_t0)


def _today_utc():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")


def _snip(text, n=160):
    return re.sub(r"\s+", " ", str(text))[:n]


def _write_if_changed(path, text):
    """既存と同じ中身なら書かない（Git の履歴を増やさない）。戻り値: True=書いた / False=同じ。"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                if f.read() == text:
                    return False
        except Exception:
            pass
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return True


def _month_rule(month_start, months, day):
    """月初の日付 → 『月数ヶ月後の day 日』（規則で決めた公表日の推定）。"""
    m = pd.Timestamp(month_start).replace(day=1) + pd.DateOffset(months=months)
    return m + pd.Timedelta(days=day - 1)


def _num_str(v):
    """数値を、実行のたびに同じ文字列になるよう整える（変化の判定用）。"""
    try:
        f = float(v)
    except Exception:
        return ""
    if f != f or f in (float("inf"), float("-inf")):
        return ""
    return format(f, ".10g")


# ------------------------------------------------------------
# (15) 産業統計（FRED）
#   IPG3344S / CAPUTLG3344S … FRB の鉱工業生産・設備稼働率（半導体・その他電子部品, NAICS 3344, 季調済）。毎月中旬
#   U34SVS / U34STI / U34SNO / U34SUO … 国勢調査局 M3（コンピュータ・電子製品, 原数値）の出荷・在庫・新規受注・受注残
#   A34SNO / A34SUO … 同・季調済（新規受注・受注残）
#   PCU334413334413 … 生産者物価（半導体・関連デバイス, 1998-12=100）
#   ※ 公表遅れの規則は『推定』（保守的）。正確な日は『初めて見えた日』のログで確かめる
#   ※ 末尾が False のものは、作った時点で FRED 上のIDを確認できていない（取れなければ NG になるだけ）
# ------------------------------------------------------------

IND_DIR = "industry_stats"
IND_FRED = {
    # FRED ID: (説明, (公表までの月数, 日), IDを確認済みか)
    "IPG3344S": ("鉱工業生産 半導体・電子部品 季調済", (1, 20), True),
    "CAPUTLG3344S": ("設備稼働率 半導体・電子部品 季調済(%)", (1, 20), True),
    "U34SVS": ("出荷額 コンピュータ・電子製品 原数値(百万ドル)", (2, 7), True),
    "U34STI": ("在庫額 コンピュータ・電子製品 原数値(百万ドル)", (2, 7), True),
    "U34SNO": ("新規受注 コンピュータ・電子製品 原数値(百万ドル)", (2, 7), True),
    "U34SUO": ("受注残 コンピュータ・電子製品 原数値(百万ドル)", (2, 7), True),
    "A34SNO": ("新規受注 コンピュータ・電子製品 季調済(百万ドル)", (1, 30), True),
    "A34SUO": ("受注残 コンピュータ・電子製品 季調済(百万ドル)", (1, 30), True),
    "PCU334413334413": ("生産者物価 半導体・関連デバイス(1998-12=100)", (1, 20), True),
    "A34SVS": ("出荷額 コンピュータ・電子製品 季調済(百万ドル)", (1, 30), False),
    "A34STI": ("在庫額 コンピュータ・電子製品 季調済(百万ドル)", (1, 30), False),
}
IND_LOG_COLS = ["Series", "ObsDate", "Value", "SeenUTC", "Backfill"]


def _fred_csv(sid):
    """FRED の1系列（fredgraph.csv）を Date 索引・Value 列で返す。"""
    r = _http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {_snip(r.text)}")
    df = pd.read_csv(io.StringIO(r.text))
    dc = "observation_date" if "observation_date" in df.columns else ("DATE" if "DATE" in df.columns else None)
    if dc is None or sid not in df.columns:
        raise RuntimeError(f"列が想定と違います {list(df.columns)[:4]} {_snip(r.text, 80)}")
    df[dc] = pd.to_datetime(df[dc])
    df = df.rename(columns={dc: "Date", sid: "Value"})[["Date", "Value"]]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"]).sort_values("Date").set_index("Date")
    if df.empty:
        raise RuntimeError("有効な行がありません")
    return df


def _update_observed_log(path, frames, today):
    """（系列, 観測月, 値）の組を『初めて見えた日』つきで足す。値が改定されたら別の行として足す。
    その系列を初めて取った回の一括分は Backfill=1（実際に見えた日ではないので、先読み防止には使えない）。"""

    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str).fillna("")
    else:
        old = pd.DataFrame(columns=IND_LOG_COLS)

    adds = []
    for sid, df in frames.items():
        have = old[old["Series"] == sid]
        known = set(zip(have["ObsDate"], have["Value"]))
        first_time = len(have) == 0
        for d, v in zip(df.index.strftime("%Y-%m-%d"), df["Value"].map(_num_str)):
            if v and (d, v) not in known:
                adds.append({"Series": sid, "ObsDate": d, "Value": v, "SeenUTC": today, "Backfill": "1" if first_time else "0"})

    if not adds:
        return 0

    merged = pd.concat([old, pd.DataFrame(adds)], ignore_index=True)
    merged = merged.sort_values(["Series", "ObsDate", "SeenUTC"], kind="stable").reset_index(drop=True)
    _write_if_changed(path, merged[IND_LOG_COLS].to_csv(index=False))
    return len(adds)


def run_block_industry():

    today = _today_utc()
    frames = {}

    for sid, (desc, (mo, dy), verified) in IND_FRED.items():

        if _x4_left() < 60:
            _extra_log("IND_TIME", False, "時間切れのため残りは次回")
            break

        try:
            df = _fred_csv(sid)
            out = df.copy()
            out["AvailableFrom"] = [_month_rule(d, mo, dy).strftime("%Y-%m-%d") for d in out.index]
            out["AvailableRule"] = f"月+{mo}の{dy}日(推定)"
            out["DataCollectedAt"] = out.index.max().strftime("%Y-%m-%d")
            _write_if_changed(os.path.join(IND_DIR, f"{sid}_historical.csv"), out.to_csv(index_label="Date"))
            frames[sid] = df
            _extra_log(f"IND_{sid}", True, f"{desc} {df.index.min().date()}〜{df.index.max().date()} {len(df)}件")
        except Exception as e:
            _extra_log(f"IND_{sid}", False, f"{desc} 取得失敗{'' if verified else '（IDを確認できていない系列）'}: {e!r}")

    if frames:
        try:
            n = _update_observed_log(os.path.join(IND_DIR, "_observed_values.csv"), frames, today)
            _extra_log("IND_SEEN_LOG", True, f"初めて見えた値のログに {n}行を追加" + ("（系列を初めて取った回の一括分は Backfill=1）" if n else ""))
        except Exception as e:
            _extra_log("IND_SEEN_LOG", False, f"ログ更新失敗: {e!r}")


# ------------------------------------------------------------
# (16) 台湾上場企業の月次売上（TWSE オープンAPI。認証不要）
#   エンドポイント /opendata/t187ap05_L … 上場会社の月次営業収入（直近の月のみ。千台湾ドル）
#   法定の届出期限は翌月10日 → AvailableFrom は『翌月11日』（推定）。過去分は遡れないので毎回少しずつ溜まる。
#   TSMC は第86回で取得した履歴（2012年〜。2012年分は2013年の前年比からの逆算で、Source に d）を種として入れる。
#   ※ 欄名はAPIの説明書の表記（営業收入-當月營收 など）に合わせ、記号を除いて照合する。実接続は未確認
# ------------------------------------------------------------

TW_DIR = "taiwan_revenue"
TW_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"
TW_COMPANIES = {
    "2330": "TSMC", "2303": "UMC", "2454": "MediaTek", "3711": "ASE", "2408": "Nanya Technology", "2344": "Winbond",
    "2337": "Macronix", "3034": "Novatek", "2379": "Realtek", "6770": "PSMC", "2449": "KYEC", "3443": "GUC", "3661": "Alchip",
    "3037": "Unimicron", "8046": "Nan Ya PCB", "3189": "Kinsus", "6415": "Silergy",
    "2317": "Hon Hai", "2382": "Quanta", "3231": "Wistron", "6669": "Wiwynn", "2356": "Inventec", "2376": "Gigabyte",
    "2345": "Accton", "2308": "Delta Electronics", "3017": "AVC", "2368": "Gold Circuit", "2360": "Chroma",
}
TW_COLS = ["Month", "Code", "Name", "Revenue_NTD_M", "PrevMonth_NTD_M", "LastYear_NTD_M", "MoM_pct", "YoY_pct",
           "YTD_NTD_M", "YTD_YoY_pct", "AvailableFrom", "FirstSeen", "Source"]
# 種（月, 売上[百万台湾ドル], 前年比[%], d=逆算）… investor.tsmc.com の年別ページ（2026-10-29 取得）
TSMC_SEED = """
2012-01,34602,,d
2012-02,33895,,d
2012-03,37119,,d
2012-04,40543,,d
2012-05,44188,,d
2012-06,43466,,d
2012-07,48558,,d
2012-08,49542,,d
2012-09,43403,,d
2012-10,49995,,d
2012-11,44286,,d
2012-12,37159,,d
2013-01,47439,37.1,
2013-02,41182,21.5,
2013-03,44134,18.9,
2013-04,50071,23.5,
2013-05,51788,17.2,
2013-06,54028,24.3,
2013-07,52103,7.3,
2013-08,55091,11.2,
2013-09,55382,27.6,
2013-10,51795,3.6,
2013-11,44330,0.1,
2013-12,49681,33.7,
2014-01,51430,8.4,
2014-02,46829,13.7,
2014-03,49956,13.2,
2014-04,61887,23.6,
2014-05,60789,17.4,
2014-06,60344,11.7,
2014-07,64925,24.6,
2014-08,69279,25.8,
2014-09,74846,35.1,
2014-10,80736,55.9,
2014-11,72275,63,
2014-12,69510,39.9,
2015-01,87120,69.4,
2015-02,62645,33.8,
2015-03,72269,44.7,
2015-04,75330,21.7,
2015-05,70155,15.4,
2015-06,59955,-0.6,
2015-07,80953,24.7,
2015-08,67038,-3.2,
2015-09,64514,-13.8,
2015-10,81743,1.2,
2015-11,63428,-12.2,
2015-12,58347,-16.1,
2016-01,70855,-18.7,
2016-02,59551,-4.9,
2016-03,73089,1.1,
2016-04,66843,-11.3,
2016-05,73576,4.9,
2016-06,81391,35.8,
2016-07,76392,-5.6,
2016-08,94311,40.7,
2016-09,89703,39,
2016-10,91085,11.4,
2016-11,93030,46.7,
2016-12,78112,33.9,
2017-01,76616,8.1,
2017-02,71423,19.9,
2017-03,85875,17.5,
2017-04,56872,-14.9,
2017-05,72796,-1.1,
2017-06,84187,3.4,
2017-07,71611,-6.3,
2017-08,91917,-2.5,
2017-09,88579,-1.3,
2017-10,94520,3.8,
2017-11,93153,0.1,
2017-12,89897,15.1,
2018-01,79741,4.1,
2018-02,64641,-9.5,
2018-03,103697,20.8,
2018-04,81870,44,
2018-05,80969,11.2,
2018-06,70438,-16.3,
2018-07,74371,3.9,
2018-08,91055,-0.9,
2018-09,94922,7.2,
2018-10,101550,7.4,
2018-11,98389,5.6,
2018-12,89831,-0.1,
2019-01,78094,-2.1,
2019-02,60889,-5.8,
2019-03,79722,-23.1,
2019-04,74694,-8.8,
2019-05,80437,-0.7,
2019-06,85868,21.9,
2019-07,84758,14,
2019-08,106118,16.5,
2019-09,102170,7.6,
2019-10,106040,4.4,
2019-11,107884,9.7,
2019-12,103313,15,
2020-01,103683,32.8,
2020-02,93394,53.4,
2020-03,113520,42.4,
2020-04,96002,28.5,
2020-05,93819,16.6,
2020-06,120878,40.8,
2020-07,105963,25,
2020-08,122878,15.8,
2020-09,127585,24.9,
2020-10,119303,12.5,
2020-11,124865,15.7,
2020-12,117365,13.6,
2021-01,126749,22.2,
2021-02,106534,14.1,
2021-03,129127,13.7,
2021-04,111315,16,
2021-05,112360,19.8,
2021-06,148471,22.8,
2021-07,124558,17.5,
2021-08,137427,11.8,
2021-09,152685,19.7,
2021-10,134539,12.8,
2021-11,148268,18.7,
2021-12,155382,32.4,
2022-01,172176,35.8,
2022-02,146933,37.9,
2022-03,171967,33.2,
2022-04,172561,55,
2022-05,185705,65.3,
2022-06,175874,18.5,
2022-07,186763,49.9,
2022-08,218132,58.7,
2022-09,208248,36.4,
2022-10,210266,56.3,
2022-11,222706,50.2,
2022-12,192560,23.9,
2023-01,200051,16.2,
2023-02,163174,11.1,
2023-03,145408,-15.4,
2023-04,147900,-14.3,
2023-05,176537,-4.9,
2023-06,156404,-11.1,
2023-07,177616,-4.9,
2023-08,188686,-13.5,
2023-09,180430,-13.4,
2023-10,243203,15.7,
2023-11,206026,-7.5,
2023-12,176300,-8.4,
2024-01,215785,7.9,
2024-02,181648,11.3,
2024-03,195211,34.3,
2024-04,236021,59.6,
2024-05,229620,30.1,
2024-06,207869,32.9,
2024-07,256953,44.7,
2024-08,250866,33,
2024-09,251873,39.6,
2024-10,314240,29.2,
2024-11,276058,34,
2024-12,278163,57.8,
2025-01,293288,35.9,
2025-02,260009,43.1,
2025-03,285957,46.5,
2025-04,349567,48.1,
2025-05,320516,39.6,
2025-06,263709,26.9,
2025-07,323166,25.8,
2025-08,335772,33.8,
2025-09,330980,31.4,
2025-10,367473,16.9,
2025-11,343614,24.5,
2025-12,335003,20.4,
2026-01,401255,36.8,
2026-02,317657,22.2,
2026-03,415191,45.2,
2026-04,410726,17.5,
2026-05,416975,30.1,
2026-06,442680,67.9,
2026-07,467580,44.7,
2026-08,514806,53.3,
2026-09,511857,54.6,
"""


def _tw_norm(k):
    return re.sub(r"[\s_\-－—–()（）%％/／]", "", str(k))


def _tw_month(s):
    """資料年月（民国年+月 '11509' など。西暦6桁 '202609' も可）→ 'YYYY-MM'。"""
    d = re.sub(r"\D", "", str(s))
    if len(d) < 4:
        raise ValueError(f"資料年月が読めません: {s!r}")
    y, m = int(d[:-2]), int(d[-2:])
    if y < 1000:
        y += 1911
    if not 1 <= m <= 12:
        raise ValueError(f"月が不正です: {s!r}")
    return f"{y:04d}-{m:02d}"


def _tw_num(v):
    s = str(v).replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return np.nan


def _tw_avail(month):
    return _month_rule(pd.Timestamp(month + "-01"), 1, 11).strftime("%Y-%m-%d")


def _tw_seed_frame():
    rows = []
    for ln in TSMC_SEED.strip().splitlines():
        p = ln.split(",")
        if len(p) < 3:
            continue
        rows.append({"Month": p[0], "Code": "2330", "Name": "TSMC", "Revenue_NTD_M": float(p[1]),
                     "YoY_pct": (float(p[2]) if p[2] else np.nan), "AvailableFrom": _tw_avail(p[0]), "FirstSeen": "",
                     "Source": ("investor.tsmc.com（2012年分は2013年の前年比からの逆算）" if (len(p) > 3 and p[3] == "d") else "investor.tsmc.com")})
    return pd.DataFrame(rows, columns=TW_COLS)


def _tw_parse(items, today):
    """APIの行（辞書のリスト）→ 表。対象会社だけ残す。"""
    rows = []
    for it in items:
        n = {_tw_norm(k): v for k, v in it.items()}
        code = str(n.get("公司代號", n.get("公司代号", ""))).strip()
        if code not in TW_COMPANIES:
            continue

        def pick(*names):
            for nm in names:
                if nm in n:
                    return _tw_num(n[nm])
            return np.nan

        month = _tw_month(n.get("資料年月", n.get("资料年月", "")))
        thousand = pick("營業收入當月營收", "营业收入当月营收")
        rows.append({
            "Month": month, "Code": code, "Name": TW_COMPANIES[code],
            "Revenue_NTD_M": thousand / 1000.0,
            "PrevMonth_NTD_M": pick("營業收入上月營收", "营业收入上月营收") / 1000.0,
            "LastYear_NTD_M": pick("營業收入去年當月營收", "营业收入去年当月营收") / 1000.0,
            "MoM_pct": pick("營業收入上月比較增減", "营业收入上月比较增减"),
            "YoY_pct": pick("營業收入去年同月增減", "营业收入去年同月增减"),
            "YTD_NTD_M": pick("累計營業收入當月累計營收", "累计营业收入当月累计营收") / 1000.0,
            "YTD_YoY_pct": pick("累計營業收入前期比較增減", "累计营业收入前期比较增减"),
            "AvailableFrom": _tw_avail(month), "FirstSeen": today, "Source": "TWSE OpenAPI t187ap05_L",
        })
    if not rows:
        raise RuntimeError("対象会社の行がありません（欄名や形式が変わった可能性）")
    return pd.DataFrame(rows, columns=TW_COLS)


def _tw_merge(path, new):
    """既存（種を含む）に足す。同じ（月, 会社）は新しい値で置き換えるが、FirstSeen は最初のものを残す。"""
    seed = _tw_seed_frame()
    if os.path.exists(path):
        old = pd.read_csv(path, dtype={"Code": str})
        old["Code"] = old["Code"].astype(str)
        base = pd.concat([old, seed], ignore_index=True).drop_duplicates(["Month", "Code"], keep="first")   # 既存を優先（種は足りない月だけ）
    else:
        base = seed
    allr = pd.concat([base, new], ignore_index=True)
    first = allr.groupby(["Month", "Code"])["FirstSeen"].transform(lambda s: next((x for x in s if isinstance(x, str) and x), ""))
    allr["FirstSeen"] = first
    allr = allr.drop_duplicates(["Month", "Code"], keep="last")
    allr = allr.sort_values(["Code", "Month"]).reset_index(drop=True)
    return allr[TW_COLS]


def run_block_taiwan():

    today = _today_utc()
    path = os.path.join(TW_DIR, "TW_MONTHLY_REVENUE_historical.csv")

    r = _http_get(TW_URL, tries=3, timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} {_snip(r.text)}")
    items = json.loads(r.content.decode("utf-8-sig"))
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"JSONの形が想定と違います {_snip(r.text)}")

    new = _tw_parse(items, today)
    merged = _tw_merge(path, new)
    _write_if_changed(path, merged.to_csv(index=False))

    tsmc = merged[merged["Code"] == "2330"]
    _extra_log("TW_REVENUE", True,
          f"今回 {len(new)}社（{new['Month'].iloc[0]}）／累計 {len(merged)}行、TSMC {tsmc['Month'].min()}〜{tsmc['Month'].max()} {len(tsmc)}か月")


# ------------------------------------------------------------
# 実行（各ブロックは独立。1つが失敗しても他は続ける）
# ------------------------------------------------------------

if GH_EXTRA_ENABLED:

    print("")
    print("=" * 70)
    print("★追加データ: GitHub Actions で無料取得できる追加データを取得中...")
    print("=" * 70)

    _GH_BLOCKS = [
        ("FRB流動性", run_block_liquidity),
        ("CFTC建玉", run_block_cftc),
        ("CBOE put/call", run_block_cboe_pc),
        ("FINRA空売り", run_block_finra),
        ("オプション板", run_block_options),
        ("保有銘柄", run_block_holdings),
        ("QQQ構成銘柄", run_block_qqq_members),
        # ---- 2026-10 追加その4（決算・産業統計まわり。連絡先もキーも不要。専用の時間予算） ----
        ("産業統計(FRED)", run_block_industry),
        ("台湾月次売上", run_block_taiwan),
    ]

    for _bn, _fn in _GH_BLOCKS:
        try:
            _fn()
        except Exception as e:
            _extra_log(f"GH_{_bn}", False, f"ブロック全体が失敗: {e!r}")

    _extra_log("GH_EXTRA_TIME", True, f"所要 {time.time() - _gh_t0:.0f}秒（上限 {GH_TIME_BUDGET_SEC}秒）")


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
    "FRB流動性（FedAssets / TGA_Wed / TGA_Daily / ON_RRP / NET_LIQUIDITY / BreakEven10Y / SOFR）/ cftc/ / cboe_pc/ / finra_shortvol/"
)
print(
    "options_snapshot/（SMH・QQQ のIV・スキュー）/ holdings_snapshot/（SMH・SOXX 保有銘柄）/ QQQ_MEMBERS_PIT"
)
print(
    "決算・産業統計: industry_stats/（FRED 半導体・電子産業）/ taiwan_revenue/（TSMC ほか月次売上）/ analyst/（決算日・予想EPSの改定・格付け・四半期財務諸表）"
)
print(
    "取得結果の一覧: EXTRA_DATA_STATUS.txt"
)

print("")
print("すべての処理が完了しました。")
