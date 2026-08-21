import yfinance as yf

data = yf.download(
    "SMH",
    period="1y",
    interval="1d",
    auto_adjust=False
)

print(data.tail())

data.to_csv("SMH_historical.csv")

print("SMHのヒストリカルデータを取得しました。")
