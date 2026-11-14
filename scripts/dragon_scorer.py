"""模块一：龙头识别打分。

按五大维度对单只股票打分（满分100）：

| 维度     | 权重 | 关键指标                                  |
|----------|------|------------------------------------------|
| 涨停质量 | 25%  | 封板时间 / 封单 / 炸板 / 回封             |
| 题材级别 | 25%  | 国家级 / 行业级 / 事件级（需用户/概念）  |
| 板块效应 | 20%  | 板块涨停家数 / 联动性 / 跟风股数          |
| 技术结构 | 15%  | 均线多头 / 量价配合 / 突破形态           |
| 市场地位 | 15%  | 连板高度 / 是否最高连板 / 先于板块启动   |

输出 JSON，含每项原始指标 + 子项得分 + 总分 + 龙头概率区间 + 结论。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import data_client as dc
import concept_heat as ch
import lhb_analyzer as lhb
from utils import (
    analyze_seal_quality, count_consecutive_limit_ups,
    is_limit_up, ma_alignment, volume_ratio,
)

# ---------------------------------------------------------------------------
# 评分常量
# ---------------------------------------------------------------------------
EARLY_SEAL_THRESHOLD = "10:00"   # 早盘封板加分阈值
LATE_SEAL_THRESHOLD = "14:00"   # 尾盘封板扣分阈值

DAILY_LOOKBACK_DAYS = 120       # 拉取日线窗口（满足 MA60）
MIN_LOOKBACK_DAYS = 65


# ---------------------------------------------------------------------------
# 维度一：涨停质量
# ---------------------------------------------------------------------------
def score_seal_quality(seal: dict, daily_today: pd.Series,
                       daily_df: pd.DataFrame) -> dict:
    """涨停质量打分（满分25）。"""
    sub = {}
    seal_time = seal.get("seal_time")
    broke_count = seal.get("broke_count", 0)
    broken = seal.get("broken", False)

    # 1) 封板时间（10分）
    if broken or not seal_time:
        sub["封板时间"] = 0
    elif seal_time < "09:45":
        sub["封板时间"] = 10
    elif seal_time < "10:30":
        sub["封板时间"] = 8
    elif seal_time < "11:30":
        sub["封板时间"] = 6
    elif seal_time < "13:30":
        sub["封板时间"] = 5
    elif seal_time < "14:00":
        sub["封板时间"] = 4
    elif seal_time < "14:45":
        sub["封板时间"] = 2
    else:
        sub["封板时间"] = 1

    # 2) 封单强度（5分，用当日量比与封板时长近似）
    v_ratio = volume_ratio(daily_df, n=5)
    sub["封单强度"] = _seal_strength_score(seal_time, v_ratio, broken)
    sub["量比"] = round(v_ratio, 2) if v_ratio else None

    # 3) 炸板次数（6分）
    if broken:
        sub["炸板控制"] = 0
    elif broke_count == 0:
        sub["炸板控制"] = 6
    elif broke_count == 1:
        sub["炸板控制"] = 4
    elif broke_count == 2:
        sub["炸板控制"] = 2
    else:
        sub["炸板控制"] = 0

    # 4) 一字/T字板加分（4分）
    if not broken and daily_today is not None:
        op = daily_today.get("open")
        lo = daily_today.get("low")
        lu = daily_today.get("limit_up")
        if lu and op and lo and abs(op - lu) < 1e-3 and abs(lo - lu) < 1e-3:
            sub["板型"] = 4  # 一字板
        elif lu and op and abs(op - lu) < 1e-3:
            sub["板型"] = 3  # T字板
        elif lu and lo and abs(lo - lu) < 1e-3:
            sub["板型"] = 2  # 开盘最低，随后封死
        else:
            sub["板型"] = 1
    else:
        sub["板型"] = 0

    total = sum(sub.get(k, 0) for k in ("封板时间", "封单强度",
                                       "炸板控制", "板型"))
    total = min(25, max(0, total))
    return {"score": total, "max": 25, "detail": sub,
            "raw": {"seal_time": seal_time, "broke_count": broke_count,
                    "broken": broken}}


def _seal_strength_score(seal_time: Optional[str],
                         v_ratio: Optional[float],
                         broken: bool) -> int:
    if broken or not seal_time:
        return 0
    # 早盘封死 + 量比适中(2~5) 满分；过大可能炸板风险
    base = 3
    if seal_time < "10:30":
        base = 5
    if v_ratio is not None:
        if 2.0 <= v_ratio <= 5.0:
            base = min(5, base)
        elif v_ratio > 8.0:
            base = max(2, base - 2)  # 量比过大，封单结构脆弱
        elif v_ratio < 1.0:
            base = max(1, base - 2)
    return base


# ---------------------------------------------------------------------------
# 维度二：题材级别（需用户/概念输入，此处给默认框架）
# ---------------------------------------------------------------------------
def score_theme_level(theme_level: Optional[str],
                      concept_name: Optional[str] = None) -> dict:
    """题材级别打分（满分25）。"""
    mapping = {
        "国家级": 25,    # 国家级政策/战略（如自主可控、新质生产力、央企改革）
        "行业级": 18,    # 行业级景气反转/产业趋势
        "事件级": 12,    # 突发事件催化（如地缘、产品发布）
        "题材不清": 6,
        None: 10,        # 未提供时给中性偏上分，AI 应追问
    }
    score = mapping.get(theme_level, 10)
    return {
        "score": score, "max": 25,
        "detail": {"题材级别": theme_level or "未提供（需追问）",
                   "所属概念": concept_name},
        "raw": {"theme_level": theme_level, "concept_name": concept_name},
        "需要追问": theme_level is None,
    }


# ---------------------------------------------------------------------------
# 维度三：板块效应
# ---------------------------------------------------------------------------
def score_sector_effect(heat: dict) -> dict:
    """板块效应打分（满分20）。"""
    luc = heat.get("limit_up_count", 0)
    broken = heat.get("broken_count", 0)
    follow = heat.get("follow_count", 0)
    median_pct = heat.get("median_pct", 0.0)

    sub = {}
    # 板块涨停家数（10分）
    if luc >= 8:
        sub["板块涨停家数"] = 10
    elif luc >= 5:
        sub["板块涨停家数"] = 8
    elif luc >= 3:
        sub["板块涨停家数"] = 6
    elif luc >= 1:
        sub["板块涨停家数"] = 3
    else:
        sub["板块涨停家数"] = 0

    # 联动性（板块涨幅中位数）（5分）
    if median_pct >= 0.05:
        sub["板块联动"] = 5
    elif median_pct >= 0.03:
        sub["板块联动"] = 4
    elif median_pct >= 0.01:
        sub["板块联动"] = 2
    else:
        sub["板块联动"] = 0

    # 跟风股数（5分）
    if follow >= 5:
        sub["跟风股数"] = 5
    elif follow >= 3:
        sub["跟风股数"] = 4
    elif follow >= 1:
        sub["跟风股数"] = 2
    else:
        sub["跟风股数"] = 0

    # 炸板股过多扣分（盘中触及涨停但未封住）
    penalty = min(3, broken // 2)
    total = max(0, sum(sub.values()) - penalty)
    total = min(20, total)
    return {"score": total, "max": 20, "detail": sub,
            "raw": {"limit_up_count": luc, "broken_count": broken,
                    "follow_count": follow, "median_pct": round(median_pct, 4),
                    "industry_name": heat.get("industry_name")}}


# ---------------------------------------------------------------------------
# 维度四：技术结构
# ---------------------------------------------------------------------------
def score_technical(daily_df: pd.DataFrame) -> dict:
    """技术结构打分（满分15）。"""
    sub = {}
    closes = daily_df.sort_values("date")["close"]
    ma_info = ma_alignment(daily_df)
    v_ratio = volume_ratio(daily_df, n=5)

    # 均线多头（8分）
    if ma_info["bullish"]:
        sub["均线多头"] = 8
    elif ma_info["ma5"] and ma_info["ma10"] and ma_info["ma5"] > ma_info["ma10"]:
        sub["均线多头"] = 4
    else:
        sub["均线多头"] = 0

    # 量价配合（4分）
    if v_ratio is not None:
        if 1.5 <= v_ratio <= 4.0:
            sub["量价配合"] = 4
        elif v_ratio > 4.0:
            sub["量价配合"] = 2  # 放量过猛，分歧信号
        elif v_ratio >= 1.0:
            sub["量价配合"] = 2
        else:
            sub["量价配合"] = 0  # 缩量涨停反成强势？此处按中性偏负
    else:
        sub["量价配合"] = 1

    # 突破近20日高点（3分）
    if len(closes) >= 21:
        high20 = daily_df.sort_values("date")["high"].iloc[-21:-1].max()
        today_high = daily_df.sort_values("date")["high"].iloc[-1]
        sub["突破20日高"] = 3 if today_high >= high20 else 0
    else:
        sub["突破20日高"] = 1

    total = min(15, max(0, sum(sub.values())))
    return {"score": total, "max": 15, "detail": sub,
            "raw": {"ma_info": {k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in ma_info.items()},
                    "量比": round(v_ratio, 2) if v_ratio else None}}


# ---------------------------------------------------------------------------
# 维度五：市场地位
# ---------------------------------------------------------------------------
def score_market_position(streak: int, market_height: Optional[int],
                          sector_first: Optional[bool]) -> dict:
    """市场地位打分（满分15）。

    streak        : 该股当前连板高度
    market_height : 全市场最高连板高度（由模块六 market_env 提供，可空）
    sector_first  : 是否先于板块启动（用户/概念热度的"龙头领涨"判断，可空）
    """
    sub = {}
    # 连板高度（8分）
    if streak >= 6:
        sub["连板高度"] = 8
    elif streak >= 4:
        sub["连板高度"] = 6
    elif streak >= 3:
        sub["连板高度"] = 5
    elif streak >= 2:
        sub["连板高度"] = 3
    else:
        sub["连板高度"] = 1

    # 是否最高连板（4分）
    if market_height is not None and streak >= market_height and streak >= 2:
        sub["最高连板"] = 4
    elif market_height is not None and streak >= market_height - 1:
        sub["最高连板"] = 2
    else:
        sub["最高连板"] = 0

    # 是否先于板块启动（3分）
    if sector_first is True:
        sub["先于板块启动"] = 3
    elif sector_first is False:
        sub["先于板块启动"] = 0
    else:
        sub["先于板块启动"] = 1  # 未知，给中性

    total = min(15, max(0, sum(sub.values())))
    return {"score": total, "max": 15, "detail": sub,
            "raw": {"streak": streak, "market_height": market_height,
                    "sector_first": sector_first}}


# ---------------------------------------------------------------------------
# 综合评分与结论
# ---------------------------------------------------------------------------
def make_verdict(total: int, probability: int) -> str:
    if total >= 80 and probability >= 75:
        return "真龙头"
    if total >= 65 and probability >= 60:
        return "候选龙头"
    if total < 50 or probability < 40:
        return "假龙头"
    return "尚需观察"


def score_to_probability(total: int) -> int:
    """将综合分数映射为龙头概率（0-100）。线性映射 + 钳制。"""
    p = round(50 + (total - 60) * 1.2)
    return max(5, min(95, p))


def main_run(symbol: str, trade_date: str,
             theme_level: Optional[str] = None,
             market_height: Optional[int] = None,
             sector_first: Optional[bool] = None) -> dict:
    # --- 1. 日线（前复权更平滑，但 limit_up 字段在 get_stock_daily 中）
    start = dc.get_prev_trade_date(trade_date, n=DAILY_LOOKBACK_DAYS)
    daily = dc.get_stock_daily(symbol=symbol, start_date=start,
                               end_date=trade_date, fields=[])
    if daily is None or daily.empty:
        return {"symbol": symbol, "error": "未取到日线数据，请检查代码或日期"}
    daily = daily.sort_values("date").reset_index(drop=True)
    today_row = daily[daily["date"].astype(str) == str(trade_date)]
    if today_row.empty:
        return {"symbol": symbol, "error": f"{trade_date} 当日无日线数据"}
    today_row = today_row.iloc[0]
    limit_up = today_row.get("limit_up")

    # --- 2. 分钟线（封板质量）
    min_df = dc.get_stock_min(symbol=symbol, start_date=trade_date,
                              end_date=trade_date, frequency="1m")
    seal = analyze_seal_quality(min_df, limit_up)

    # --- 3. 板块效应
    heat_res = ch.analyze(symbol, trade_date, level="L1")
    heat = heat_res["heat"] or {}
    industry_name = heat.get("industry_name")

    # --- 4. 龙虎榜（近 3 日）
    lhb_res = lhb.analyze(symbol, trade_date, lookback_days=3)

    # --- 5. 连板高度
    streak = count_consecutive_limit_ups(daily)

    # --- 评分
    s1 = score_seal_quality(seal, today_row, daily)
    s2 = score_theme_level(theme_level, industry_name)
    s3 = score_sector_effect(heat)
    s4 = score_technical(daily)
    s5 = score_market_position(streak, market_height, sector_first)

    total = s1["score"] + s2["score"] + s3["score"] + s4["score"] + s5["score"]
    probability = score_to_probability(total)
    verdict = make_verdict(total, probability)

    # 关键优势 / 主要风险
    adv, risk = _derive_adv_risk(s1, s2, s3, s4, s5, lhb_res, streak)

    return {
        "symbol": symbol,
        "name": today_row.get("name"),
        "trade_date": str(trade_date),
        "close": float(today_row["close"]),
        "limit_up": float(limit_up) if limit_up else None,
        "is_limit_up_today": bool(is_limit_up(today_row)),
        "streak": streak,
        "industry_name": industry_name,
        "scores": {
            "涨停质量": s1, "题材级别": s2, "板块效应": s3,
            "技术结构": s4, "市场地位": s5,
        },
        "total_score": total,
        "max_score": 100,
        "probability": probability,
        "probability_band": f"{max(0, probability-5)}%-{min(100, probability+5)}%",
        "verdict": verdict,
        "advantages": adv,
        "risks": risk,
        "lhb": {
            "on_lhb": lhb_res.get("on_lhb"),
            "seat_score": lhb_res.get("seat_quality", {}).get("seat_score"),
            "top5_net": lhb_res.get("seat_quality", {}).get("top5_net"),
            "has_institution": lhb_res.get("seat_quality", {}).get("has_institution"),
        },
        "raw_sector_heat": heat,
        "needs_followup": s2.get("需要追问") or market_height is None,
    }


def _derive_adv_risk(s1, s2, s3, s4, s5, lhb_res, streak) -> tuple[list, list]:
    adv, risk = [], []
    if s1["detail"].get("封板时间", 0) >= 8:
        adv.append("早盘快速封板，资金抢筹意图明显")
    if s5["detail"].get("最高连板", 0) >= 4:
        adv.append(f"连板高度{streak}板，处于市场最高位梯队")
    if s4["detail"].get("均线多头", 0) == 8:
        adv.append("均线多头排列，趋势结构健康")
    if lhb_res.get("seat_quality", {}).get("has_institution") and \
       lhb_res.get("seat_quality", {}).get("institution_net", 0) > 0:
        adv.append("龙虎榜机构净买入，机构资金加持")

    if s1["detail"].get("炸板控制", 6) == 0:
        risk.append("当日炸板未回封或多次炸板，封单结构脆弱")
    if s3["detail"].get("板块涨停家数", 0) <= 3:
        risk.append("板块联动性弱，缺乏板块共振")
    if s4["detail"].get("量价配合", 0) == 2 and s4["raw"].get("量比", 0) and s4["raw"]["量比"] > 4:
        risk.append("量比过高（>4），存在高位分歧风险")
    if lhb_res.get("seat_quality", {}).get("top5_net", 0) < 0:
        risk.append("龙虎榜前五席位净卖出，主力有出货迹象")
    if s2.get("需要追问"):
        risk.append("题材级别未确认，需追问以避免误判")
    return adv[:2], risk[:2]


def main():
    ap = argparse.ArgumentParser(description="龙头识别打分")
    ap.add_argument("--symbol", required=True, help="股票代码，如 000001.SZ")
    ap.add_argument("--date", default=None, help="交易日 YYYYMMDD")
    ap.add_argument("--theme", default=None,
                    choices=["国家级", "行业级", "事件级", "题材不清"],
                    help="题材级别（不传则标注需追问）")
    ap.add_argument("--market-height", type=int, default=None,
                    help="全市场最高连板（由 market_env 提供）")
    ap.add_argument("--sector-first", default=None,
                    choices=["true", "false"],
                    help="是否先于板块启动")
    args = ap.parse_args()

    trade_date = args.date or dc.get_last_trade_date("SH")
    sector_first = None
    if args.sector_first:
        sector_first = (args.sector_first == "true")

    result = main_run(args.symbol, trade_date,
                      theme_level=args.theme,
                      market_height=args.market_height,
                      sector_first=sector_first)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
