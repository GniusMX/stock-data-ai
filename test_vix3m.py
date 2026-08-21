import yfinance as yf

print("VIX3Mの取得テストを開始します")

ticker = yf.Ticker("^VIX3M")

data = ticker.history(
    start="2007-12-04",
    end="2026-08-23",
    interval="1d",
    auto_adjust=False
)

print("")
print("取得件数:", len(data))

if len(data) > 0:
    print("")
    print("最初の5件:")
    print(data.head())

    print("")
    print("最後の5件:")
    print(data.tail())

    print("")
    print("最古の日付:", data.index.min())
    print("最新の日付:", data.index.max())

    data.to_csv("VIX3M_test.csv")

    print("")
    print("VIX3M_test.csv を作成しました")

else:
    print("")
    print("VIX3Mのデータを取得できませんでした")
