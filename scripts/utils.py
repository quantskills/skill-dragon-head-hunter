"""龙头战法通用工具：涨停识别、炸板、连板高度、封板时间、均线。"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# 涨停价容差：处理浮点误差。ST/主板/创业板/科创板涨跌停规则不同，
# 但 PandaData 已返回 limit_up 字段，只需比较 close 是否触及。
_LIMIT_TOL = 1e-3


def is_limit_up(row: pd.Series) -> bool:
    """收盘等于涨停价视为当日涨停（封板成功）。"""
    lu = row.get("limit_up")
    if lu is None or pd.isna(lu) or lu == 0:
        return False
    return abs(row["close"] - lu) <= _LIMIT_TOL


def is_broken_limit(row: pd.Series) -> bool:
    """当日盘中触及涨停但未封住收盘：high >= limit_up 且 close < limit_up。"""
    lu = row.get("limit_up")
    if lu is None or pd.isna(lu) or lu == 0:
        return False
    touched = row["high"] >= lu - _LIMIT_TOL
    sealed = abs(row["close"] - lu) <= _LIMIT_TOL
    return bool(touched and not sealed)


def is_touched_limit(row: pd.Series) -> bool:
    """盘中是否曾触及涨停（不论最终是否封住）。"""
    lu = row.get("limit_up")
    if lu is None or pd.isna(lu) or lu == 0:
        return False
    return row["high"] >= lu - _LIMIT_TOL


def count_consecutive_limit_ups(daily_df: pd.DataFrame) -> int:
    """从最新交易日向前数连续涨停天数（连板高度）。

    daily_df 应按日期升序排列，含 limit_up / close 字段。
    返回：从最后一行起向前的连续涨停天数。
    """
    if daily_df is None or daily_df.empty:
        return 0
    df = daily_df.sort_values("date").reset_index(drop=True)
    n = 0
    for _, row in df.iloc[::-1].iterrows():
        if is_limit_up(row):
            n += 1
        else:
            break
    return n


def find_seal_time(min_df: pd.DataFrame, limit_up: Optional[float]) -> Optional[str]:
    """从分钟线定位首次封板时间（HH:MM）。

    判定：分钟 close 触及 limit_up 即视为该分钟封板。
    返回该分钟的 'HH:MM'，若未封板返回 None。
    """
    if min_df is None or min_df.empty or limit_up is None:
        return None
    df = min_df.copy()
    if "minute" in df.columns:
        df = df.sort_values(["date", "minute"]).reset_index(drop=True)
    elif "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        return None
    mask = (df["close"] - limit_up) >= -_LIMIT_TOL
    if not mask.any():
        return None
    first = df.loc[mask.idxmax()]
    return _hhmm(first)


def analyze_seal_quality(min_df: pd.DataFrame, limit_up: Optional[float]) -> dict:
    """综合分析封板质量。

    返回:
      seal_time: 首次封板时间 'HH:MM' 或 None
      reseal_count: 全天炸板后回封次数
      broke_count: 全天炸板次数（封住后又被打开）
      last_seal_time: 最终封死的时间
      broken: 收盘是否炸板（未封住）
    """
    out = {"seal_time": None, "reseal_count": 0, "broke_count": 0,
           "last_seal_time": None, "broken": False}
    if min_df is None or min_df.empty or limit_up is None:
        return out
    df = min_df.copy()
    if "minute" in df.columns:
        df = df.sort_values(["date", "minute"]).reset_index(drop=True)
    elif "datetime" in df.columns:
        df = df.sort_values("datetime").reset_index(drop=True)
    else:
        return out

    sealed = (df["close"] - limit_up) >= -_LIMIT_TOL
    if not sealed.any():
        out["broken"] = True
        return out

    out["seal_time"] = _hhmm(df.loc[sealed.idxmax()])
    # 计算 broke_count：封住 → 跌破 → 再次封住 视为 1 次回封
    state = "pre"
    broke_count = 0
    reseal_count = 0
    for s in sealed:
        if state == "pre" and s:
            state = "sealed"
        elif state == "sealed" and not s:
            state = "broke"
            broke_count += 1
        elif state == "broke" and s:
            state = "sealed"
            reseal_count += 1
    out["broke_count"] = broke_count
    out["reseal_count"] = reseal_count
    out["last_seal_time"] = _hhmm(df[sealed].iloc[-1]) if sealed.any() else None
    out["broken"] = not sealed.iloc[-1]
    return out


def _hhmm(row: pd.Series) -> Optional[str]:
    """统一提取 'HH:MM'。兼容 minute='HHMMSS' 与 datetime='YYYY-MM-DD HH:MM:SS'。"""
    if "minute" in row and isinstance(row["minute"], str):
        m = row["minute"]
        if len(m) >= 4:                       # "HHMMSS" 或 "HHMM"
            return f"{m[:2]}:{m[2:4]}"
        return m
    if "datetime" in row and isinstance(row["datetime"], str) and len(row["datetime"]) >= 16:
        return row["datetime"][11:16]
    return None


def compute_ma(close_series: pd.Series, n: int) -> Optional[float]:
    """计算最近 N 日均线值。close_series 升序。"""
    if close_series is None or len(close_series) < n:
        return None
    return float(close_series.tail(n).mean())


def ma_alignment(daily_df: pd.DataFrame) -> dict:
    """均线多头排列判断：MA5>MA10>MA20>MA60。"""
    closes = daily_df.sort_values("date")["close"]
    ma5 = compute_ma(closes, 5)
    ma10 = compute_ma(closes, 10)
    ma20 = compute_ma(closes, 20)
    ma60 = compute_ma(closes, 60)
    bullish = None
    if all(m is not None for m in (ma5, ma10, ma20, ma60)):
        bullish = bool(ma5 > ma10 > ma20 > ma60)
    return {"ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60, "bullish": bullish}


def volume_ratio(daily_df: pd.DataFrame, n: int = 5) -> Optional[float]:
    """当日量 / 前 N 日均量。"""
    if daily_df is None or len(daily_df) < n + 1:
        return None
    df = daily_df.sort_values("date").reset_index(drop=True)
    today = df["volume"].iloc[-1]
    avg_prev = df["volume"].iloc[-n - 1:-1].mean()
    if not avg_prev or avg_prev == 0:
        return None
    return float(today / avg_prev)


def turnover_rate(daily_df: pd.DataFrame) -> Optional[float]:
    """最近一日换手率近似：当日成交量 / 前 N 日均量 (proxy)。"""
    return volume_ratio(daily_df, n=5)
