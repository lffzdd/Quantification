"""
Day 3：回测框架搭建
==================

今天的目标：
1. 搭建一个面向对象的、可复用的回测引擎
2. 支持 A 股交易费用（佣金 + 印花税 + 滑点）
3. 自动计算六大风险指标 (Day 2 复习)
4. 自动生成绩效报告图表
5. 用 Day 1 的 SMA 策略作为示例运行

运行方式：
    uv run python src/day3_backtest_engine.py

核心架构：
    Strategy (抽象基类)
        └── SMAStrategy (具体策略)
    BacktestEngine (回测引擎)
        ├── run()           → 执行回测
        ├── _apply_costs()  → 计算交易费用
        ├── _calc_metrics() → 计算绩效指标
        └── plot_report()   → 生成可视化报告
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 第一部分：Strategy 抽象基类
# ============================================================
# 为什么用抽象基类？
#   - 定义"接口"：所有策略必须实现 generate_signals() 方法
#   - 面向接口编程：BacktestEngine 不需要知道具体策略是什么
#   - Day 4~7 的策略都会继承这个基类，直接接入回测
#
# 程序员类比：这就像 Java 的 Interface 或 Go 的 Interface
# 在 Python 中用 ABC (Abstract Base Class) 实现


class Strategy(ABC):
    """策略抽象基类：所有策略都必须继承此类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称，用于报告展示。"""
        ...

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        根据行情数据生成交易信号。

        参数:
            df: 包含 OHLCV 列的 DataFrame (Open, High, Low, Close, Volume)
                索引为 DatetimeIndex

        返回:
            pd.Series: 信号序列，与 df 同索引
                1  = 买入/持有（做多）
                0  = 空仓
               -1  = 做空（A股暂不支持，预留接口）

        注意:
            - 不需要在这里做 shift(1)，引擎会自动处理
            - 可以使用 df 中的任何列来生成信号
        """
        ...


# ============================================================
# 第二部分：具体策略实现 —— SMA 双均线策略
# ============================================================
# 这是 Day 1 学过的策略，现在用 OOP 的方式重新实现
# 以后每个新策略都这样写：继承 Strategy，实现 generate_signals


class SMAStrategy(Strategy):
    """
    双均线交叉策略 (Simple Moving Average Crossover)

    逻辑：
        - 当短期均线 > 长期均线 → 买入（趋势向上）
        - 当短期均线 ≤ 长期均线 → 空仓（趋势向下）

    参数：
        short_window: 短期均线天数（默认 20）
        long_window:  长期均线天数（默认 50）
    """

    def __init__(self, short_window: int = 20, long_window: int = 50):
        self.short_window = short_window
        self.long_window = long_window

    @property
    def name(self) -> str:
        return f"SMA({self.short_window}/{self.long_window})"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sma_short = df["Close"].rolling(window=self.short_window).mean()
        sma_long = df["Close"].rolling(window=self.long_window).mean()

        # 初始为 0（空仓），短均线上穿长均线时设为 1（持有）
        signal = pd.Series(0, index=df.index)
        signal[sma_short > sma_long] = 1
        return signal


# ============================================================
# 第三部分：交易费用配置
# ============================================================
# 用 dataclass 来定义费用参数，清晰且不易出错


@dataclass
class CostConfig:
    """
    A 股交易费用配置

    属性:
        commission_rate: 佣金费率（双向收取）
            - 默认 0.025% = 万 2.5（目前市场普遍水平）
            - 某些券商可以低到万 1
        stamp_tax_rate: 印花税（仅卖出时收取）
            - 2023年8月28日起为 0.05%（之前是 0.1%）
        slippage_rate: 滑点
            - 模拟市场冲击，默认 0.01%
            - 大资金或小盘股应该设更高
        min_commission: 最低佣金（元）
            - 券商规定每笔交易最低收取 5 元佣金
            - 本框架暂不实现（需要引入资金量概念），但请知道这个规则
    """
    commission_rate: float = 0.00025  # 万 2.5
    stamp_tax_rate: float = 0.0005   # 万 5 (卖出)
    slippage_rate: float = 0.0001    # 万 1

    @property
    def buy_cost(self) -> float:
        """买入总成本率 = 佣金 + 滑点"""
        return self.commission_rate + self.slippage_rate

    @property
    def sell_cost(self) -> float:
        """卖出总成本率 = 佣金 + 印花税 + 滑点"""
        return self.commission_rate + self.stamp_tax_rate + self.slippage_rate

    def __str__(self) -> str:
        return (
            f"佣金: {self.commission_rate:.4%} | "
            f"印花税: {self.stamp_tax_rate:.4%} | "
            f"滑点: {self.slippage_rate:.4%}"
        )


# ============================================================
# 第四部分：回测引擎（核心！）
# ============================================================


class BacktestEngine:
    """
    回测引擎：接收策略和数据，执行回测，输出绩效。

    用法：
        strategy = SMAStrategy(20, 50)
        engine = BacktestEngine(strategy, df)
        result = engine.run()
        engine.plot_report()
    """

    TRADING_DAYS = 242  # A 股年交易日

    def __init__(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        cost_config: Optional[CostConfig] = None,
        risk_free_rate: float = 0.02,
    ):
        """
        参数:
            strategy: 策略实例（继承自 Strategy）
            data: 行情数据，必须包含 Close 列，索引为 DatetimeIndex
            cost_config: 交易费用配置，默认使用 A 股标准费率
            risk_free_rate: 年无风险利率，默认 2%
        """
        self.strategy = strategy
        self.data = data.copy()
        self.cost_config = cost_config or CostConfig()
        self.risk_free_rate = risk_free_rate
        self.result: Optional[pd.DataFrame] = None
        self.metrics: dict = {}
        self.trades: list[dict] = []

    def run(self) -> dict:
        """
        执行回测主流程。

        流程：
        1. 策略生成信号
        2. 信号延迟一天（避免未来函数）
        3. 计算交易费用
        4. 计算策略收益
        5. 计算绩效指标

        返回:
            dict: 绩效指标字典
        """
        df = self.data

        # ---- Step 1: 生成信号 ----
        # 策略只负责产出"原始信号"，不需要关心延迟和费用
        df["Signal"] = self.strategy.generate_signals(df)

        # ---- Step 2: 信号延迟一天 ----
        # 这是回测中最关键的一步！
        # 今天收盘产生的信号 → 明天才能执行
        # shift(1) 把信号整体向后移一天
        #
        # 举例：
        #   Signal:   [0, 0, 1, 1, 0]  ← 第3天决定买入
        #   Position: [NaN, 0, 0, 1, 1] ← 第4天才真正持有
        df["Position"] = df["Signal"].shift(1)

        # ---- Step 3: 检测交易 ----
        # 仓位变化 = 今天的仓位 - 昨天的仓位
        # diff > 0 → 买入（仓位增加）
        # diff < 0 → 卖出（仓位减少）
        # diff = 0 → 无交易（持仓不变）
        df["Position_Change"] = df["Position"].diff()

        # ---- Step 4: 计算交易费用 ----
        df["Cost"] = self._calculate_costs(df)

        # ---- Step 5: 计算收益 ----
        # 基准收益：买入持有的日收益率
        df["Daily_Return"] = df["Close"].pct_change()

        # 策略收益 = 仓位 × 日收益 - 交易费用
        df["Strategy_Return"] = (
            df["Position"] * df["Daily_Return"] - df["Cost"]
        )

        # 累计净值
        df["Benchmark_Equity"] = (1 + df["Daily_Return"]).cumprod()
        df["Strategy_Equity"] = (1 + df["Strategy_Return"]).cumprod()

        # ---- Step 6: 记录交易 ----
        self._record_trades(df)

        # ---- Step 7: 计算绩效指标 ----
        self.result = df.dropna(subset=["Strategy_Return"])
        self.metrics = self._calculate_metrics()

        return self.metrics

    def _calculate_costs(self, df: pd.DataFrame) -> pd.Series:
        """
        根据仓位变化计算交易费用。

        逻辑：
        - 仓位增加（买入）：扣 buy_cost
        - 仓位减少（卖出）：扣 sell_cost
        - 仓位不变：无费用

        为什么用 abs(Position_Change)？
        因为费用与交易量成比例。
        如果从 0 变到 1（全仓买入），交易量 = 1 × 持仓市值
        费用 = 1 × buy_cost_rate
        """
        cost = pd.Series(0.0, index=df.index)
        pos_change = df["Position_Change"]

        # 买入：仓位增加
        buy_mask = pos_change > 0
        cost[buy_mask] = pos_change[buy_mask].abs() * self.cost_config.buy_cost

        # 卖出：仓位减少
        sell_mask = pos_change < 0
        cost[sell_mask] = pos_change[sell_mask].abs() * self.cost_config.sell_cost

        return cost

    def _record_trades(self, df: pd.DataFrame) -> None:
        """记录所有交易，方便后续分析。"""
        self.trades = []
        trade_mask = df["Position_Change"].fillna(0) != 0

        for date, row in df[trade_mask].iterrows():
            action = "BUY" if row["Position_Change"] > 0 else "SELL"
            self.trades.append({
                "Date": date,
                "Action": action,
                "Price": row["Close"],
                "Position_Change": row["Position_Change"],
                "Cost": row["Cost"],
            })

    def _calculate_metrics(self) -> dict:
        """
        计算六大风险指标（Day 2 复习！）。

        这里把 Day 2 的所有指标整合到一个方法里，
        以后任何策略跑完都自动算出完整绩效。
        """
        df = self.result
        returns = df["Strategy_Return"]
        bench_returns = df["Daily_Return"]
        n_days = len(returns)
        td = self.TRADING_DAYS

        # ---- 策略绩效 ----
        # 年化收益率（几何平均）
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + total_return) ** (td / n_days) - 1

        # 年化波动率
        annual_vol = returns.std() * np.sqrt(td)

        # 夏普比率
        rf_daily = self.risk_free_rate / td
        excess = returns - rf_daily
        sharpe = excess.mean() / excess.std() * np.sqrt(td) if excess.std() > 0 else 0

        # 最大回撤
        equity = (1 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()

        # 索提诺比率
        downside = excess[excess < 0]
        downside_std = downside.std() * np.sqrt(td) if len(downside) > 0 else np.nan
        sortino = (annual_return - self.risk_free_rate) / downside_std if downside_std and downside_std > 0 else 0

        # 卡尔玛比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # ---- 基准绩效（买入持有） ----
        bench_total = (1 + bench_returns).prod() - 1
        bench_annual = (1 + bench_total) ** (td / n_days) - 1

        # ---- 交易统计 ----
        n_trades = len(self.trades)
        total_cost = df["Cost"].sum()

        # 胜率（用配对交易计算）
        win_rate = self._calculate_win_rate()

        return {
            "策略名称": self.strategy.name,
            "回测区间": f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
            "交易天数": n_days,
            # -- 收益 --
            "总收益率": total_return,
            "年化收益率": annual_return,
            "基准年化收益率": bench_annual,
            "超额年化收益率": annual_return - bench_annual,
            # -- 风险 --
            "年化波动率": annual_vol,
            "最大回撤": max_drawdown,
            # -- 风险调整 --
            "夏普比率": sharpe,
            "索提诺比率": sortino,
            "卡尔玛比率": calmar,
            # -- 交易 --
            "交易次数": n_trades,
            "总交易费用": total_cost,
            "胜率": win_rate,
        }

    def _calculate_win_rate(self) -> float:
        """
        计算胜率：盈利交易次数 / 总交易次数。

        逻辑：将买卖配对，计算每组配对的收益。
        """
        if len(self.trades) < 2:
            return 0.0

        wins = 0
        total_pairs = 0
        buy_price = None

        for t in self.trades:
            if t["Action"] == "BUY":
                buy_price = t["Price"]
            elif t["Action"] == "SELL" and buy_price is not None:
                total_pairs += 1
                if t["Price"] > buy_price:
                    wins += 1
                buy_price = None

        return wins / total_pairs if total_pairs > 0 else 0.0

    def print_report(self) -> None:
        """打印文字版绩效报告。"""
        if not self.metrics:
            print("❌ 请先调用 run() 执行回测！")
            return

        m = self.metrics
        print("\n" + "=" * 65)
        print(f"📊 回测报告：{m['策略名称']}")
        print(f"   {m['回测区间']}（{m['交易天数']} 个交易日）")
        print(f"   费用设置：{self.cost_config}")
        print("=" * 65)

        print("\n📈 收益指标")
        print(f"   总收益率:       {m['总收益率']:>10.2%}")
        print(f"   年化收益率:     {m['年化收益率']:>10.2%}")
        print(f"   基准年化收益:   {m['基准年化收益率']:>10.2%}")
        print(f"   超额年化收益:   {m['超额年化收益率']:>10.2%}")

        print("\n📉 风险指标")
        print(f"   年化波动率:     {m['年化波动率']:>10.2%}")
        print(f"   最大回撤:       {m['最大回撤']:>10.2%}")

        print("\n⚖️  风险调整指标")
        print(f"   夏普比率:       {m['夏普比率']:>10.3f}")
        print(f"   索提诺比率:     {m['索提诺比率']:>10.3f}")
        print(f"   卡尔玛比率:     {m['卡尔玛比率']:>10.3f}")

        print("\n🔄 交易统计")
        print(f"   交易次数:       {m['交易次数']:>10d}")
        print(f"   总交易费用:     {m['总交易费用']:>10.4%}")
        print(f"   胜率:           {m['胜率']:>10.2%}")
        print("=" * 65)

    def plot_report(self, save_path: Optional[str] = None) -> None:
        """
        生成 4 合 1 绩效报告图表。

        子图1: 净值曲线（策略 vs 基准）
        子图2: 回撤曲线
        子图3: 月度收益热力图
        子图4: 交易标记图
        """
        if self.result is None:
            print("❌ 请先调用 run() 执行回测！")
            return

        df = self.result
        m = self.metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(
            f"回测报告: {m['策略名称']}    "
            f"Sharpe={m['夏普比率']:.2f}  "
            f"MaxDD={m['最大回撤']:.2%}  "
            f"年化={m['年化收益率']:.2%}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        # ---- 子图1: 净值曲线对比 ----
        ax1 = axes[0, 0]
        ax1.plot(df.index, df["Strategy_Equity"], label="策略", color="#2196F3", linewidth=1.5)
        ax1.plot(df.index, df["Benchmark_Equity"], label="买入持有", color="#9E9E9E",
                 linewidth=1, linestyle="--", alpha=0.8)
        ax1.axhline(y=1, color="gray", linestyle=":", alpha=0.3)
        ax1.fill_between(
            df.index,
            df["Strategy_Equity"],
            df["Benchmark_Equity"],
            where=df["Strategy_Equity"] >= df["Benchmark_Equity"],
            alpha=0.1, color="green", label="策略领先"
        )
        ax1.fill_between(
            df.index,
            df["Strategy_Equity"],
            df["Benchmark_Equity"],
            where=df["Strategy_Equity"] < df["Benchmark_Equity"],
            alpha=0.1, color="red", label="基准领先"
        )
        ax1.set_title("净值曲线（策略 vs 买入持有）")
        ax1.set_ylabel("净值")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ---- 子图2: 回撤曲线 ----
        ax2 = axes[0, 1]
        equity = df["Strategy_Equity"]
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        ax2.fill_between(drawdown.index, 0, drawdown.values, color="#F44336", alpha=0.4)
        ax2.plot(drawdown.index, drawdown.values, color="#F44336", linewidth=0.8)
        ax2.axhline(y=m["最大回撤"], color="#D32F2F", linestyle="--",
                     label=f"最大回撤: {m['最大回撤']:.2%}")
        ax2.set_title("回撤曲线")
        ax2.set_ylabel("回撤幅度")
        ax2.legend(loc="lower right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ---- 子图3: 月度收益柱状图 ----
        ax3 = axes[1, 0]
        monthly_ret = df["Strategy_Return"].resample("ME").apply(
            lambda x: (1 + x).prod() - 1
        )
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in monthly_ret]
        ax3.bar(monthly_ret.index, monthly_ret.values, width=20, color=colors, alpha=0.8)
        ax3.axhline(y=0, color="gray", linewidth=0.5)
        ax3.set_title("月度收益")
        ax3.set_ylabel("月收益率")
        ax3.grid(True, alpha=0.3)

        # ---- 子图4: 价格图 + 交易标记 ----
        ax4 = axes[1, 1]
        ax4.plot(df.index, df["Close"], color="#607D8B", linewidth=0.8, label="收盘价")

        # 标记买卖点
        buy_trades = [t for t in self.trades if t["Action"] == "BUY"]
        sell_trades = [t for t in self.trades if t["Action"] == "SELL"]

        if buy_trades:
            ax4.scatter(
                [t["Date"] for t in buy_trades],
                [t["Price"] for t in buy_trades],
                marker="^", color="#4CAF50", s=40, zorder=5, label=f"买入 ({len(buy_trades)}次)"
            )
        if sell_trades:
            ax4.scatter(
                [t["Date"] for t in sell_trades],
                [t["Price"] for t in sell_trades],
                marker="v", color="#F44336", s=40, zorder=5, label=f"卖出 ({len(sell_trades)}次)"
            )

        ax4.set_title("交易标记")
        ax4.set_ylabel("价格")
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\n📊 报告已保存到 {save_path}")

        plt.show()


# ============================================================
# 第五部分：数据加载工具函数
# ============================================================

def load_stock_data(
    symbol: str,
    start_date: str = "20220101",
    end_date: str = "20260101",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    加载 A 股数据（带缓存）。

    参数:
        symbol: 股票代码（如 "600519"）
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        adjust: 复权方式 ("qfq" 前复权 / "hfq" 后复权 / "" 不复权)

    返回:
        pd.DataFrame: 标准化后的行情数据 (OHLCV)
    """
    cache_path = Path(f"src/{symbol}_{adjust}.parquet")

    if cache_path.exists():
        print(f"📂 从缓存加载 {symbol}...")
        df = pd.read_parquet(cache_path)
    else:
        print(f"🌐 下载 {symbol} 数据...")
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
        df.to_parquet(cache_path)
        print(f"💾 已缓存到 {cache_path}")

    # 标准化列名
    df = df.rename(columns={
        "日期": "Date", "开盘": "Open", "收盘": "Close",
        "最高": "High", "最低": "Low", "成交量": "Volume",
        "成交额": "Amount", "换手率": "Turnover",
        "振幅": "Amplitude", "涨跌幅": "Change_Pct",
        "涨跌额": "Change_Amt",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


# ============================================================
# 第六部分：主程序 —— 运行示例
# ============================================================

if __name__ == "__main__":

    # ---- 1. 加载数据 ----
    print("=" * 65)
    print("🚀 Day 3：回测框架演示")
    print("=" * 65)

    df = load_stock_data("600519")
    print(f"\n数据量: {len(df)} 行，范围: {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}")

    # ---- 2. 创建策略 ----
    strategy = SMAStrategy(short_window=20, long_window=50)
    print(f"策略: {strategy.name}")

    # ---- 3. 运行回测（对比：有费用 vs 无费用）----

    # 3a. 有交易费用（真实场景）
    print("\n" + "-" * 40)
    print("📌 场景 A：包含交易费用（真实场景）")
    print("-" * 40)
    engine_with_cost = BacktestEngine(
        strategy=strategy,
        data=df,
        cost_config=CostConfig(),  # 默认 A 股费率
    )
    engine_with_cost.run()
    engine_with_cost.print_report()

    # 3b. 无交易费用（理想场景）
    print("\n" + "-" * 40)
    print("📌 场景 B：不含交易费用（理想场景）")
    print("-" * 40)
    engine_no_cost = BacktestEngine(
        strategy=strategy,
        data=df,
        cost_config=CostConfig(
            commission_rate=0,
            stamp_tax_rate=0,
            slippage_rate=0,
        ),
    )
    engine_no_cost.run()
    engine_no_cost.print_report()

    # ---- 4. 对比费用影响 ----
    m1 = engine_with_cost.metrics
    m2 = engine_no_cost.metrics

    print("\n" + "=" * 65)
    print("⚡ 费用影响对比")
    print("=" * 65)
    print(f"                    有费用        无费用        差异")
    print(f"   年化收益率:   {m1['年化收益率']:>10.2%}   {m2['年化收益率']:>10.2%}   {m1['年化收益率'] - m2['年化收益率']:>10.2%}")
    print(f"   夏普比率:     {m1['夏普比率']:>10.3f}   {m2['夏普比率']:>10.3f}   {m1['夏普比率'] - m2['夏普比率']:>10.3f}")
    print(f"   总费用侵蚀:  {m1['总交易费用']:>10.4%}")
    print("=" * 65)

    # ---- 5. 打印交易记录（前 10 条）----
    print(f"\n📋 交易记录（共 {len(engine_with_cost.trades)} 笔，显示前 10 笔）：")
    print(f"   {'日期':　<12s}  {'操作':　<5s}  {'价格':>10s}  {'费用':>10s}")
    print("   " + "-" * 50)
    for t in engine_with_cost.trades[:10]:
        action_str = "🟢 买入" if t["Action"] == "BUY" else "🔴 卖出"
        print(f"   {t['Date'].strftime('%Y-%m-%d')}  {action_str}  {t['Price']:>10.2f}  {t['Cost']:>10.4%}")

    # ---- 6. 生成图表报告 ----
    engine_with_cost.plot_report(save_path="src/day3_report.png")

    print("\n✅ Day 3 完成！")
    print("💡 核心收获：")
    print("   1. 搭建了可复用的 Strategy + BacktestEngine 框架")
    print("   2. 理解了 A 股交易费用对策略的影响")
    print("   3. 后续只需 继承 Strategy 类 就能接入新策略")
    print("\n🔜 Day 4 预告：动量策略 — 用今天的框架运行！")
