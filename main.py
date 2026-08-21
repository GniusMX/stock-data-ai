import yfinance as yf

# 取得する銘柄・指数
tickers = {
    "SMH": "SMH",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M"
}

# 1銘柄ずつ取得してCSV保存
for name, ticker in tickers.items():

    print(f"{name} のデータを取得中...")

    data = yf.download(
        ticker,
        period="1y",
        interval="1d",
        auto_adjust=False
    )

    print(data.tail())

    filename = f"{name}_historical.csv"
    data.to_csv(filename)

    print(f"{filename} を保存しました。")

print("すべてのデータ取得が完了しました。")
