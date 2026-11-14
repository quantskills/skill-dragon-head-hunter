"""模块六：市场环境评估。

通过全市场单日日线扫描，输出：
  limit_up_count       : 涨停家数
  broken_count         : 炸板家数（盘中触及涨停未封住）
  touched_count        : 盘中触及涨停家数
  broken_rate          : 炸板率 = broken / touched
  max_streak           : 全市场最高连板高度
  top_streak_stocks    : 连板高度前 N 名（供模块一调用 market_height）
  limit_down_count     : 跌停家数
  total_amount         : 全市场成交额
  up_ratio / down_ratio: 上涨/下跌家数比
  sentiment            : 情绪档位（积极进攻/谨慎参与/防守为主/空仓观望）
  main_lines           : 当前最强主线（行业维度统计涨停家数 top3）

判断规则（量化）：
  limit_up >= 80 且 broken_rate <= 15%  → 积极进攻
  limit_up >= 40 且 broken_rate <= 25%  → 谨慎参与
  limit_up >= 20 或 broken_rate <= 40% → 防守为主
  否则                                  → 空仓观望

注：盘中调用时今日 EOD 未生成，会自动回退到最近一个已收盘的交易日。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import data_client as dc
from utils import is_limit_up, is_broken_limit, is_touched_limit, \
    count_consecutive_limit_ups

STREAK_LOOKBACK = 10          # 计算连板高度时回看天数
TOP_N_STREAK = 10            # 输出连板高度前 N
TOP_SECTOR_N = 3             # 输出主线行业数


def fetch_market_daily(target_date: str) -> tuple[pd.DataFrame, str]:
    """获取 target_date 当日全市场日线（含 limit_up 字段）。

    流程：
      1. 调用 get_trade_list 拿该日在售股票列表
      2. 若 target_date 为今日盘中（在售列表为空），自动回退到前一交易日
      3. 用拿到的股票列表批量调 get_stock_daily 获取当日 EOD
      4. 若仍为空（极少数情况），用 rt_daily 的 symbol 列表再试一次

    返回 (DataFrame, 实际使用的交易日)
    """
    # 1. 在售股票列表
    trade_list = dc.get_trade_list(date=target_date, exchange="SH")
    if trade_list is None or trade_list.empty:
        # 今日盘中数据未生成，回退到前一个交易日
        target_date = dc.get_prev_trade_date(target_date, n=1)
        trade_list = dc.get_trade_list(date=target_date, exchange="SH")

    syms = trade_list["symbol"].dropna().unique().tolist() if trade_list is not None and not trade_list.empty else []

    # 2. 批量获取当日 EOD
    df = dc.get_stock_daily(symbol=syms, start_date=target_date,
                            end_date=target_date, fields=[])

    # 3. 兜底：用 rt_daily 的 symbol 列表重试
    if (df is None or df.empty) and not syms:
        rt = dc.get_stock_rt_daily(symbol=[])
        syms = rt["symbol"].dropna().unique().tolist()
        df = dc.get_stock_daily(symbol=syms, start_date=target_date,
                                end_date=target_date, fields=[])

    if df is None or df.empty:
        return pd.DataFrame(), target_date

    # 仅保留当日数据
    df = df[df["date"].astype(str) == str(target_date)].copy()
    return df, target_date


def compute_basics(daily: pd.DataFrame) -> dict:
    """基础涨跌停/炸板统计。"""
    if daily.empty:
        return {"limit_up_count": 0, "broken_count": 0, "touched_count": 0,
                "broken_rate": 0.0, "limit_down_count": 0,
                "total_amount": 0.0, "up_ratio": 0.0, "down_ratio": 0.0,
                "up_count": 0, "down_count": 0, "flat_count": 0,
                "total_stocks": 0}

    limit_up_mask = daily.apply(is_limit_up, axis=1)
    touched_mask = daily.apply(is_touched_limit, axis=1)
    broken_mask = daily.apply(is_broken_limit, axis=1)
    ld_mask = daily.apply(
        lambda r: bool(r.get("limit_down") and abs(r["close"] - r["limit_down"]) <= 1e-3),
        axis=1,
    )

    pct = (daily["close"] / daily["pre_close"] - 1.0)
    pct = pct.dropna()
    up = int((pct > 0).sum())
    down = int((pct < 0).sum())
    flat = int((pct == 0).sum())
    total_ud = up + down
    return {
        "limit_up_count": int(limit_up_mask.sum()),
        "broken_count": int(broken_mask.sum()),
        "touched_count": int(touched_mask.sum()),
        "broken_rate": round(float(broken_mask.sum()) / touched_mask.sum(), 4)
                       if touched_mask.sum() else 0.0,
        "limit_down_count": int(ld_mask.sum()),
        "total_amount": float(daily["amount"].sum()) if "amount" in daily else 0.0,
        "up_count": up, "down_count": down, "flat_count": flat,
        "up_ratio": round(up / total_ud, 4) if total_ud else 0.0,
        "down_ratio": round(down / total_ud, 4) if total_ud else 0.0,
        "total_stocks": int(len(daily)),
    }


def compute_streaks(limit_up_symbols: list, trade_date: str,
                    lookback_days: int = STREAK_LOOKBACK) -> dict:
    """对当日涨停股回看 lookback_days 天，计算每只连板高度，取最大值。"""
    if not limit_up_symbols:
        return {"max_streak": 0, "top_streak_stocks": []}
    start = dc.get_prev_trade_date(trade_date, n=lookback_days - 1)
    df = dc.get_stock_daily(symbol=limit_up_symbols, start_date=start,
                            end_date=trade_date, fields=[])
    if df is None or df.empty:
        return {"max_streak": 0, "top_streak_stocks": []}
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    results = []
    for sym, g in df.groupby("symbol"):
        results.append({"symbol": sym,
                        "streak": count_consecutive_limit_ups(g.reset_index(drop=True))})
    results.sort(key=lambda x: x["streak"], reverse=True)
    return {
        "max_streak": results[0]["streak"] if results else 0,
        "top_streak_stocks": results[:TOP_N_STREAK],
    }


def compute_main_lines(limit_up_symbols: list, trade_date: str) -> list:
    """统计涨停股所属申万一级行业分布，取涨停家数 top3 作为主线。"""
    if not limit_up_symbols:
        return []
    industry_counter = Counter()
    for sym in limit_up_symbols:
        try:
            ind_df = dc.get_stock_industry(stock_symbol=sym, level="L1")
            if ind_df is not None and not ind_df.empty:
                name = ind_df.iloc[0].get("industry_name") \
                       or ind_df.iloc[0].get("parent_l1_name")
                if name:
                    industry_counter[name] += 1
        except Exception:
            continue
    return [{"industry": n, "limit_up_count": c}
            for n, c in industry_counter.most_common(TOP_SECTOR_N)]


def assess_sentiment(basics: dict, streaks: dict) -> tuple[str, str]:
    """根据量化阈值返回情绪档位 + 操作建议。"""
    lu = basics["limit_up_count"]
    br = basics["broken_rate"]
    ms = streaks["max_streak"]
    if lu >= 80 and br <= 0.15:
        return "积极进攻", f"涨停{lu}家, 炸板率{br:.0%}, 最高{ms}连板, 赚钱效应强"
    if lu >= 40 and br <= 0.25:
        return "谨慎参与", f"涨停{lu}家, 炸板率{br:.0%}, 情绪中性偏暖"
    if lu >= 20 or br <= 0.40:
        return "防守为主", f"涨停{lu}家, 炸板率{br:.0%}, 赚钱效应差"
    return "空仓观望", f"涨停{lu}家, 炸板率{br:.0%}, 情绪冰点或炸板潮"


def analyze(target_date: str | None = None) -> dict:
    """分析指定交易日的市场环境。

    target_date 为 None 时默认取最新交易日；若当日 EOD 未就绪会自动回退。
    """
    requested_date = target_date or dc.get_last_trade_date("SH")
    daily, used_date = fetch_market_daily(requested_date)
    if daily.empty:
        return {"requested_date": str(requested_date),
                "trade_date": str(used_date),
                "error": "未取到当日全市场日线（可能停牌日或数据未就绪）"}

    basics = compute_basics(daily)
    limit_up_symbols = daily[daily.apply(is_limit_up, axis=1)]["symbol"].tolist()

    streaks = compute_streaks(limit_up_symbols, used_date)
    main_lines = compute_main_lines(limit_up_symbols, used_date)
    sentiment, reason = assess_sentiment(basics, streaks)

    return {
        "requested_date": str(requested_date),
        "trade_date": str(used_date),
        "fallback_used": str(requested_date) != str(used_date),
        "basics": basics,
        "streaks": streaks,
        "main_lines": main_lines,
        "sentiment": sentiment,
        "reason": reason,
    }


def main():
    ap = argparse.ArgumentParser(description="市场环境评估")
    ap.add_argument("--date", default=None, help="交易日 YYYYMMDD，默认最新已收盘日")
    args = ap.parse_args()
    result = analyze(args.date)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
