import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 1. 获取数据 (Data)
# 程序员视角：这就是调用 API 获取 JSON/CSV 并转为 DataFrame
path = Path("AAPL.parquet")
if path.exists() and path.is_file():
    df = pd.read_parquet(path)
else:
    print("下载数据")
    df = yf.download("AAPL", start="2022-01-01", end="2025-01-01")
    df.to_parquet("AAPL.parquet")

# 2. 策略逻辑 (Strategy) - 向量化操作，避免 for 循环
# 逻辑：计算 20日均线 和 50日均线
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["SMA_50"] = df["Close"].rolling(window=50).mean()

# 生成信号：
# 1 代表持有 (20日线 > 50日线)
# 0 代表空仓 (20日线 <= 50日线)
df["Signal"] = 0
df.loc[df["SMA_20"] > df["SMA_50"], "Signal"] = 1

# 3. 计算收益 (Backtest)
# 核心逻辑：今天的收益取决于昨天的信号 (shift(1))
# Log Return 是为了方便累加，这里用简单收益率演示
df["Daily_Return"] = df["Close"].pct_change()
df["Strategy_Return"] = df["Daily_Return"] * df["Signal"].shift(1)

# 4. 结果可视化 (Visualization)
# 计算累计收益
df["Buy_Hold_Cum"] = (1 + df["Daily_Return"]).cumprod()
df["Strategy_Cum"] = (1 + df["Strategy_Return"]).cumprod()

print(f"买入持有总收益: {df['Buy_Hold_Cum'].iloc[-1]:.2f}")
print(f"策略总收益:     {df['Strategy_Cum'].iloc[-1]:.2f}")

# 画图
df[["Buy_Hold_Cum", "Strategy_Cum"]].plot(figsize=(10, 6))
plt.title("Moving Average Strategy vs Buy & Hold")
plt.show()
