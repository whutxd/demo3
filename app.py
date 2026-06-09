"""


A股股票回测系统 - 方正电机 sz002196
使用腾讯财经接口获取数据，双均线交叉策略回测，Streamlit 可视化展示
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
from datetime import datetime, timedelta
import requests
import io


# ────────────────── 页面配置 ──────────────────
st.set_page_config(
    page_title="A股回测系统",
    page_icon="📈",
    layout="wide",
)


# ────────────────── 数据获取 ──────────────────
@st.cache_data(ttl=3600)
def fetch_tencent_stock_data(symbol: str = "sz002196", days: int = 90) -> pd.DataFrame:
    """
    从腾讯财经接口获取A股日线数据
    接口文档: http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sz002196,day,,,90,qfq
    """
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {
        "param": f"{symbol},day,,,{days},qfq",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != 0:
        raise ValueError(f"接口返回错误: {data.get('msg', '未知错误')}")

    # 数据结构: data[symbol][qfqday] -> list of [date, open, close, high, low, vol]
    stock_data = data.get("data", {})
    symbol_data = stock_data.get(symbol, {})
    kline = symbol_data.get("qfqday", symbol_data.get("day", []))

    if not kline:
        raise ValueError(f"未能获取到 {symbol} 的K线数据，请检查股票代码是否正确")

    records = []
    for item in kline:
        if not isinstance(item, (list, dict)):
            continue
        if isinstance(item, dict):
            date = item.get("date", "")
            open_p = item.get("open", 0)
            close = item.get("close", 0)
            high = item.get("high", 0)
            low = item.get("low", 0)
            volume = item.get("vol", 0)
        else:
            date = item[0]
            open_p = float(item[1])
            close = float(item[2])
            high = float(item[3])
            low = float(item[4])
            volume = float(item[5]) if len(item) > 5 else 0
        records.append({
            "date": date,
            "open": open_p,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("K线数据为空")

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    # 确保数值类型正确
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["close"], inplace=True)
    return df


# ────────────────── 策略逻辑 ──────────────────
def double_ma_strategy(df: pd.DataFrame, fast_period: int = 5, slow_period: int = 20) -> pd.DataFrame:
    """双均线交叉策略"""
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(window=fast_period).mean()
    df["ma_slow"] = df["close"].rolling(window=slow_period).mean()

    # 信号：金叉买入(1)，死叉卖出(-1)，无信号(0)
    df["signal"] = 0
    df.loc[df["ma_fast"] > df["ma_slow"], "signal"] = 1
    df.loc[df["ma_fast"] < df["ma_slow"], "signal"] = -1

    # 交易动作：持仓变化
    df["action"] = 0
    df.loc[df["signal"].diff() == 2, "action"] = 1   # 金叉 → 买入
    df.loc[df["signal"].diff() == -2, "action"] = -1  # 死叉 → 卖出

    # 回测收益曲线
    df["returns"] = df["close"].pct_change()
    # 持有策略收益：前一天的信号决定今天是否持仓
    df["hold_returns"] = df["signal"].shift(1) * df["returns"]
    df["strategy_equity"] = (1 + df["hold_returns"]).cumprod()
    df["buy_hold_equity"] = (1 + df["returns"]).cumprod()

    return df


# ────────────────── 指标计算 ──────────────────
def calc_metrics(df: pd.DataFrame) -> dict:
    """计算回测核心指标"""
    df = df.dropna()
    if df.empty:
        return {}

    strat_ret = (df["strategy_equity"].iloc[-1] - 1) * 100
    buyhold_ret = (df["buy_hold_equity"].iloc[-1] - 1) * 100
    daily_returns = df["hold_returns"].dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # 最大回撤
    cummax = df["strategy_equity"].cummax()
    drawdown = (df["strategy_equity"] - cummax) / cummax
    max_dd = drawdown.min() * 100

    # 交易次数
    trades = abs(df["action"]).sum()

    return {
        "strategy_return": strat_ret,
        "buyhold_return": buyhold_ret,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "trade_count": int(trades),
    }


# ────────────────── 主程序 ──────────────────
st.title("📈 A股股票回测系统")

# 侧边栏参数设置（放在前面定义，后续才能引用 symbol）
with st.sidebar:
    st.header("⚙️ 参数设置")
    new_symbol = st.text_input("股票代码", "sz002196")
    fast_ma = st.slider("快均线周期", 3, 30, 5)
    slow_ma = st.slider("慢均线周期", 5, 60, 20)
    fetch_days = st.slider("获取天数", 30, 365, 90)

    # 检测股票代码是否变化，变化则清除缓存
    if "last_symbol" in st.session_state and st.session_state.last_symbol != new_symbol:
        st.cache_data.clear()
        st.session_state.dirty = False

    symbol = new_symbol
    st.session_state.last_symbol = symbol

    st.markdown("---")
    st.info(
        "数据来源：腾讯财经接口\n"
        "策略：双均线交叉 (金叉买入 / 死叉卖出)\n"
        f"股票：{symbol}"
    )

    if st.button("🔄 获取数据并回测", type="primary"):
        st.session_state.dirty = True

# 动态显示当前股票代码
st.markdown(f"**{symbol}** · 腾讯财经接口 · 双均线交叉策略")

if not st.session_state.get("dirty", False):
    st.info("👆 点击左侧 **获取数据并回测** 按钮开始回测")
    st.stop()

# ────────────────── 数据获取 + 回测 ──────────────────
try:
    df = fetch_tencent_stock_data(symbol, fetch_days)
except Exception as e:
    st.error(f"❌ 数据获取失败: {e}")
    st.stop()

if df.empty:
    st.warning("⚠️ 未获取到数据")
    st.stop()

st.subheader(f"📊 数据概览")
c1, c2, c3, c4 = st.columns(4)
c1.metric("交易日", f"{len(df)} 天")
c2.metric("最新收盘价", f"{df['close'].iloc[-1]:.3f}")
c3.metric("区间最高", f"{df['high'].max():.3f}")
c4.metric("区间最低", f"{df['low'].min():.3f}")

st.caption(f"数据区间：{df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')}")
# st.dataframe(df.tail(10), use_container_width=True)
st.dataframe(df, use_container_width=True)
# df

# ────────────────── 策略回测 ──────────────────
df = double_ma_strategy(df, fast_ma, slow_ma)
metrics = calc_metrics(df)

# 从 df 中提取买卖信号行
buy_signals = df[df["action"] == 1]
sell_signals = df[df["action"] == -1]

# ────────────────── 图表展示 ──────────────────
st.subheader("📉 K线图 + 买卖信号")

# 构造 mplfinance 的 addplot，标注买卖点
buy_ap = []
sell_ap = []
if not buy_signals.empty:
    # mplfinance scatter 需要全长度 Series，非信号点填 NaN
    b = pd.Series(np.nan, index=df.index)
    b[buy_signals.index] = buy_signals["low"] * 0.95
    buy_ap = mpf.make_addplot(b, type='scatter', marker='v', color='green', markersize=100)
if not sell_signals.empty:
    s = pd.Series(np.nan, index=df.index)
    s[sell_signals.index] = sell_signals["high"] * 1.05
    sell_ap = mpf.make_addplot(s, type='scatter', marker='^', color='red', markersize=100)

apds = []
# 添加均线
for color, col_name in [("#e74c3c", "ma_fast"), ("#3498db", "ma_slow")]:
    apds.append(mpf.make_addplot(df[col_name], color=color, width=1.2))

if buy_ap:
    apds.append(buy_ap)
if sell_ap:
    apds.append(sell_ap)

fig, axes = mpf.plot(
    df,
    type="candle",
    style="charles",
    addplot=apds if apds else None,
    mav=(fast_ma, slow_ma),
    volume=True,
    figsize=(14, 8),
    returnfig=True,
    title="",
    ylabel="Price",
    ylabel_lower="Volume",
)
plt.legend(loc="upper left", fontsize=8)
st.pyplot(fig)

# ────────────────── 收益曲线对比 ──────────────────
st.subheader("📊 策略收益 vs 买入持有")
fig2, ax2 = plt.subplots(figsize=(14, 4))
df["strategy_equity"].plot(ax=ax2, label=f"双均线策略 (收益 {metrics['strategy_return']:.2f}%)", linewidth=1.5)
df["buy_hold_equity"].plot(ax=ax2, label="买入持有 (收益 {buyhold_ret:.2f}%)".format(buyhold_ret=metrics["buyhold_return"]), linewidth=1.5, alpha=0.6)
ax2.set_title("累计收益对比", fontsize=13)
ax2.set_xlabel("")
ax2.set_ylabel("累计收益率")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# ────────────────── 回测指标 ──────────────────
st.subheader("📋 回测绩效指标")
if metrics:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("策略累计收益", f"{metrics['strategy_return']:+.2f}%")
    c2.metric("买入持有收益", f"{metrics['buyhold_return']:+.2f}%")
    c3.metric("夏普比率", f"{metrics['sharpe_ratio']:.3f}")
    c4.metric("最大回撤", f"{metrics['max_drawdown']:.2f}%")
    c5.metric("交易次数", metrics["trade_count"])

    # 收益对比柱状图
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    bar_labels = ["策略累计收益", "买入持有收益"]
    bar_values = [metrics["strategy_return"], metrics["buyhold_return"]]
    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in bar_values]
    bars = ax3.bar(bar_labels, bar_values, color=colors, width=0.5)
    for bar, val in zip(bars, bar_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (1 if val >= 0 else -3),
                 f"{val:+.2f}%", ha="center", fontsize=12, fontweight="bold")
    ax3.set_title("收益对比", fontsize=12)
    ax3.axhline(y=0, color="gray", linewidth=0.5)
    st.pyplot(fig3)

# ────────────────── 交易明细 ──────────────────
st.subheader("📝 交易明细")
trades_df = df[df["action"] != 0][["close", "ma_fast", "ma_slow", "action"]]
if not trades_df.empty:
    trades_df["type"] = trades_df["action"].map({1: "🟢 买入", -1: "🔴 卖出"})
    trades_df_display = trades_df[["type", "close", "ma_fast", "ma_slow"]]
    st.dataframe(trades_df_display, use_container_width=True)
else:
    st.info("回测期间内未触发交易信号")

# 底部备注
st.caption(
    "⚠️ 免责声明：本工具仅供学习和研究使用，不构成任何投资建议。"
    "历史回测结果不代表未来表现，股市有风险，投资需谨慎。"
)

# 标记完成
st.session_state.dirty = False
