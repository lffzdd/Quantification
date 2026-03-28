import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats

ticker = "AAPL"  # 你可以换成 "SPY"、"MSFT"、"TSLA" 等
n_days = 100  # 取最近 100 个交易日收益

# 1) 拉历史数据：使用固定日期范围确保结果可复现
df = yf.download(
    ticker,
    start="2025-06-01",
    end="2026-01-28",
    interval="1d",
    auto_adjust=True,
    progress=False,
)
# auto_adjust=True 会用复权后的价格（更适合算收益）

# 2) 用收盘价算简单日收益 r_t = P_t/P_{t-1}-1
close = df["Close"].dropna()
rets = (
    close.pct_change().dropna()
)  # pandas 的 pct_change 就是常用写法 :contentReference[oaicite:3]{index=3}

# 3) 取最近 n_days 个交易日收益
sample = rets.tail(n_days).to_numpy().flatten()  # 确保是 1D 数组
if len(sample) < n_days:
    raise ValueError(f"数据不够：只拿到 {len(sample)} 条收益")

# 4) 单样本 t 检验：H0: mean = 0
t_stat, p_two_sided = stats.ttest_1samp(sample, popmean=0.0)
t_stat = float(t_stat)
p_two_sided = float(p_two_sided)

# 5) 输出结果
mean = float(np.mean(sample))
std = float(np.std(sample, ddof=1))

print(f"Ticker: {ticker}")
print(f"Sample size: {len(sample)}")
print(f"Mean daily return: {mean:.6f}")
print(f"Std daily return:  {std:.6f}")
print(f"t-stat: {t_stat:.3f}")
print(f"p-value (two-sided): {p_two_sided:.4f}")
