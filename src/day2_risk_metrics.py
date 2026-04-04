"""
Day 2：收益率与风险度量
=====================

今天的目标：
1. 用 akshare 获取 A 股数据（替代之前的 yfinance）
2. 计算简单收益率和对数收益率
3. 计算六大核心风险指标：
   - 年化收益率
   - 年化波动率
   - 夏普比率 (Sharpe Ratio)
   - 最大回撤 (Maximum Drawdown)
   - 索提诺比率 (Sortino Ratio)
   - 卡尔玛比率 (Calmar Ratio)
4. 可视化：净值曲线、回撤曲线、滚动波动率

运行方式：
    uv run python src/day2_risk_metrics.py
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 0. 中文字体配置（matplotlib 显示中文）
# ============================================================
plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows 黑体
plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示


# ============================================================
# 1. 获取 A 股数据
# ============================================================
# akshare 是一个免费的 A 股数据源，不需要注册或 API Key
# 接口：ak.stock_zh_a_hist()
#   - symbol：股票代码（纯数字，如 "600519"）
#   - period：周期（daily / weekly / monthly）
#   - start_date / end_date：日期范围，格式 YYYYMMDD
#   - adjust：复权方式（"" 不复权, "qfq" 前复权, "hfq" 后复权）
#
# 什么是前复权？
#   股票如果分红或送股，股价会除权导致跳空。
#   前复权 = 以最新价格为基准，向前调整历史价格，使价格曲线连续。
#   做量化回测 **必须** 用复权数据，否则收益率计算会因为除权跳空而出错。

SYMBOL = "600519"  # 贵州茅台
SYMBOL_NAME = "贵州茅台"
START_DATE = "20220101"
END_DATE = "20260101"

# 缓存机制：第一次下载后保存到本地，避免重复请求
cache_path = Path(f"src/{SYMBOL}.parquet")

if cache_path.exists():
    print(f"📂 从本地缓存加载 {SYMBOL} 数据...")
    df = pd.read_parquet(cache_path)
else:
    print(f"🌐 从网络下载 {SYMBOL} ({SYMBOL_NAME}) 数据...")
    df = ak.stock_zh_a_hist(
        symbol=SYMBOL,
        period="daily",
        start_date=START_DATE,
        end_date=END_DATE,
        adjust="qfq",  # 前复权
    )
    df.to_parquet(cache_path)
    print(f"💾 已保存到 {cache_path}")

# 看看数据长什么样
print("\n📊 数据预览（前 5 行）：")
print(df.head())
print(f"\n数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
print(f"时间范围: {df['日期'].iloc[0]} ~ {df['日期'].iloc[-1]}")

# ============================================================
# 2. 数据预处理
# ============================================================
# akshare 返回的列名是中文，我们转换为英文方便后续处理
# 同时把日期列设为索引

df = df.rename(
    columns={
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
        "成交额": "Amount",
        "振幅": "Amplitude",
        "涨跌幅": "Change_Pct",
        "涨跌额": "Change_Amt",
        "换手率": "Turnover",
    }
)
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

# ============================================================
# 3. 计算收益率
# ============================================================

# --- 3.1 简单收益率 (Simple Return) ---
# 公式: R_t = P_t / P_{t-1} - 1
# pandas 的 pct_change() 就是计算这个
df["Simple_Return"] = df["Close"].pct_change()

# --- 3.2 对数收益率 (Log Return) ---
# 公式: r_t = ln(P_t / P_{t-1})
# 对数收益率的优势：多日收益率可以直接相加
df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

# 去掉第一行的 NaN（因为第一天没有前一天的数据）
df = df.dropna(subset=["Simple_Return"])

print("\n📈 收益率计算完成，看看前几行：")
print(df[["Close", "Simple_Return", "Log_Return"]].head(10))

# 验证：当收益率很小时，简单收益率 ≈ 对数收益率
diff = (df["Simple_Return"] - df["Log_Return"]).abs()
print(f"\n两种收益率的平均差异: {diff.mean():.6f}")
print(f"最大差异: {diff.max():.6f}")
print("→ 日度级别差异很小，但涨跌停板（±10%）时差异会更明显")


# ============================================================
# 4. 六大风险指标
# ============================================================

# 关键常量
TRADING_DAYS = 242  # A 股一年约 242 个交易日
RISK_FREE_RATE = 0.02  # 无风险利率，取 2%（近年国债收益率水平）

returns = df["Simple_Return"]  # 后续计算统一用简单收益率


# --- 4.1 年化收益率 ---
# 方法：几何平均
# 公式：Annual_Return = (∏(1 + R_t))^(242/n) - 1
# 这相当于：把总收益率开 n/242 次方，换算到年
total_return = (1 + returns).prod() - 1  # 总收益率
n_days = len(returns)
annual_return = (1 + total_return) ** (TRADING_DAYS / n_days) - 1

print("\n" + "=" * 60)
print("📊 风险指标计算结果")
print("=" * 60)
print(f"\n【1】年化收益率: {annual_return:.2%}")
print(f"    └─ 总收益率: {total_return:.2%}（{n_days} 个交易日）")
print(f"    └─ 公式: (1 + {total_return:.4f})^(242/{n_days}) - 1")


# --- 4.2 年化波动率 ---
# 公式：σ_annual = σ_daily × √242
# 为什么是 √242？因为假设日收益率独立同分布：
#   Var(年收益) = 242 × Var(日收益)
#   Std(年收益) = √242 × Std(日收益)
daily_vol = returns.std()
annual_vol = daily_vol * np.sqrt(TRADING_DAYS)

print(f"\n【2】年化波动率: {annual_vol:.2%}")
print(f"    └─ 日波动率: {daily_vol:.4%}")
print(f"    └─ 公式: {daily_vol:.6f} × √242 = {annual_vol:.4f}")


# --- 4.3 夏普比率 (Sharpe Ratio) ⭐ ---
# 公式：Sharpe = (R_p - R_f) / σ_p
# 但实际中更常用日度计算再年化：
#   Sharpe = mean(daily_excess_return) / std(daily_excess_return) × √242
rf_daily = RISK_FREE_RATE / TRADING_DAYS  # 日无风险利率
excess_returns = returns - rf_daily
sharpe = excess_returns.mean() / excess_returns.std(ddof=1) * np.sqrt(TRADING_DAYS)

print(f"\n【3】夏普比率 (Sharpe): {sharpe:.3f}")
print(f"    └─ 日均超额收益: {excess_returns.mean():.6f}")
print(f"    └─ 超额收益标准差: {excess_returns.std():.6f}")
if sharpe > 2:
    print("    └─ 评价: 🌟 非常优秀")
elif sharpe > 1:
    print("    └─ 评价: ✅ 不错")
elif sharpe > 0.5:
    print("    └─ 评价: ⚠️ 一般")
else:
    print("    └─ 评价: ❌ 较差")


# --- 4.4 最大回撤 (Maximum Drawdown) ⭐ ---
# 计算步骤：
# 1) 构建净值曲线（假设初始净值为 1）
# 2) 计算到每个时间点的历史最高净值（running max）
# 3) 当前净值与最高净值的差 / 最高净值 = 当前回撤
# 4) 取最大回撤
cumulative = (1 + returns).cumprod()  # 净值曲线
running_max = cumulative.cummax()  # 历史最高净值
drawdown = (cumulative - running_max) / running_max  # 回撤序列
max_drawdown = drawdown.min()  # 最大回撤（注意是负数）
max_dd_date = drawdown.idxmin()  # 最大回撤发生的日期

# 找出最大回撤的起始日期（峰值日期）
peak_date = cumulative[:max_dd_date].idxmax()

print(f"\n【4】最大回撤 (MaxDD): {max_drawdown:.2%}")
print(f"    └─ 峰值日期: {peak_date.strftime('%Y-%m-%d')} (净值: {cumulative[peak_date]:.4f})")
print(f"    └─ 谷底日期: {max_dd_date.strftime('%Y-%m-%d')} (净值: {cumulative[max_dd_date]:.4f})")
print(f"    └─ 含义: 如果你在峰值买入，最多亏损 {abs(max_drawdown):.2%}")


# --- 4.5 索提诺比率 (Sortino Ratio) ---
# 与 Sharpe 的区别：分母只用 "下行波动率"（负收益的标准差）
# 原因：上涨的波动不是"风险"，只有下跌才是真风险
downside_returns = excess_returns[excess_returns < 0]
downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS)
sortino = (annual_return - RISK_FREE_RATE) / downside_std

print(f"\n【5】索提诺比率 (Sortino): {sortino:.3f}")
print(f"    └─ 下行波动率（年化）: {downside_std:.2%}")
print(f"    └─ 下行天数: {len(downside_returns)} / {len(returns)} "
      f"({len(downside_returns) / len(returns):.1%})")
print(f"    └─ vs Sharpe ({sharpe:.3f}): ", end="")
if sortino > sharpe:
    print("Sortino 更高 → 下跌波动比上涨波动小，好事！")
else:
    print("Sortino 更低 → 下跌波动比上涨波动大，需注意")


# --- 4.6 卡尔玛比率 (Calmar Ratio) ---
# 公式：Calmar = 年化收益率 / |最大回撤|
# 含义：每承受 1% 的最大回撤，能赚多少年化收益
calmar = annual_return / abs(max_drawdown)

print(f"\n【6】卡尔玛比率 (Calmar): {calmar:.3f}")
print(f"    └─ 公式: {annual_return:.2%} / {abs(max_drawdown):.2%} = {calmar:.3f}")
if calmar > 3:
    print("    └─ 评价: 🌟 非常优秀")
elif calmar > 1:
    print("    └─ 评价: ✅ 不错")
else:
    print("    └─ 评价: ⚠️ 收益不够覆盖回撤风险")


# ============================================================
# 5. 汇总表格
# ============================================================
print("\n" + "=" * 60)
print(f"📋 {SYMBOL_NAME} ({SYMBOL}) 风险指标汇总")
print("=" * 60)

summary = {
    "年化收益率": f"{annual_return:.2%}",
    "年化波动率": f"{annual_vol:.2%}",
    "最大回撤": f"{max_drawdown:.2%}",
    "夏普比率": f"{sharpe:.3f}",
    "索提诺比率": f"{sortino:.3f}",
    "卡尔玛比率": f"{calmar:.3f}",
}

for k, v in summary.items():
    print(f"  {k:　<8s}  {v}")


# ============================================================
# 6. 可视化
# ============================================================
# 创建一个包含 4 个子图的大图

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"{SYMBOL_NAME} ({SYMBOL}) 风险分析仪表盘",
    fontsize=16,
    fontweight="bold",
    y=0.98,
)

# --- 子图1：价格与净值曲线 ---
ax1 = axes[0, 0]
ax1.plot(cumulative.index, cumulative.values, color="#2196F3", linewidth=1.2)
ax1.fill_between(
    cumulative.index,
    1,
    cumulative.values,
    alpha=0.15,
    color="#2196F3",
)
ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
ax1.set_title("净值曲线（初始 = 1）", fontsize=12)
ax1.set_ylabel("净值")
ax1.grid(True, alpha=0.3)

# --- 子图2：回撤曲线 ---
ax2 = axes[0, 1]
ax2.fill_between(
    drawdown.index,
    0,
    drawdown.values,
    color="#F44336",
    alpha=0.4,
)
ax2.plot(drawdown.index, drawdown.values, color="#F44336", linewidth=0.8)
ax2.axhline(y=max_drawdown, color="#D32F2F", linestyle="--", linewidth=1,
            label=f"最大回撤: {max_drawdown:.2%}")
ax2.set_title("回撤曲线", fontsize=12)
ax2.set_ylabel("回撤幅度")
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# --- 子图3：滚动波动率（20日） ---
ax3 = axes[1, 0]
rolling_vol = returns.rolling(window=20).std() * np.sqrt(TRADING_DAYS)
ax3.plot(rolling_vol.index, rolling_vol.values, color="#FF9800", linewidth=1)
ax3.axhline(y=annual_vol, color="gray", linestyle="--", alpha=0.7,
            label=f"整体年化波动率: {annual_vol:.2%}")
ax3.set_title("20日滚动年化波动率", fontsize=12)
ax3.set_ylabel("年化波动率")
ax3.legend(loc="upper right")
ax3.grid(True, alpha=0.3)

# --- 子图4：收益率分布直方图 ---
ax4 = axes[1, 1]
ax4.hist(returns, bins=80, color="#9C27B0", alpha=0.7, edgecolor="white", density=True)

# 叠加正态分布曲线作对比
x = np.linspace(returns.min(), returns.max(), 200)
from scipy.stats import norm
ax4.plot(x, norm.pdf(x, returns.mean(), returns.std()),
         color="#E91E63", linewidth=2, label="正态分布拟合")
ax4.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
ax4.set_title("日收益率分布", fontsize=12)
ax4.set_xlabel("日收益率")
ax4.set_ylabel("概率密度")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("src/day2_dashboard.png", dpi=150, bbox_inches="tight")
print("\n📊 图表已保存到 src/day2_dashboard.png")
plt.show()


# ============================================================
# 7. 额外知识：偏度和峰度
# ============================================================
# 这两个指标告诉我们收益率分布与"正态分布"的差异

skewness = returns.skew()
kurtosis = returns.kurtosis()  # pandas 返回的是超额峰度

print("\n" + "=" * 60)
print("📐 收益率分布特征（进阶）")
print("=" * 60)
print(f"  偏度 (Skewness):  {skewness:.3f}")
if skewness < 0:
    print("    └─ 负偏 → 左尾更长 → 极端亏损比极端盈利更常见 ⚠️")
elif skewness > 0:
    print("    └─ 正偏 → 右尾更长 → 极端盈利比极端亏损更常见 ✅")
else:
    print("    └─ 接近对称")

print(f"  超额峰度 (Excess Kurtosis): {kurtosis:.3f}")
if kurtosis > 0:
    print(f"    └─ 尖峰厚尾 → 极端事件比正态分布预测的更频繁 ⚠️")
    print(f"    └─ 这就是为什么金融中不能完全依赖正态分布假设")
else:
    print(f"    └─ 薄尾 → 极端事件比较少见")

print("\n✅ Day 2 完成！")
print("💡 下一步: Day 3 将搭建一个面向对象的回测框架")
