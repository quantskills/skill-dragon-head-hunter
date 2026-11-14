"""龙虎榜数据分析。

为模块一（市场地位/资金面）与模块三（持有与卖出）提供龙虎榜证据：

- 汇总买卖双方前5席位金额
- 判别机构席位净买卖方向
- 识别"游资接力"特征（席位净买入集中度）
- 输出用于评分的量化指标

PandaData 龙虎榜类型常用枚举：
  G0007 日涨幅偏离值达7%   T0020 日换手率达20%
  G0006 日跌幅偏离值达7%   T0021 日换手率达20%（ST/大盘股）
  G0005 日振幅值达15%      G0008 连续三个交易日内累计涨幅达20%
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import data_client as dc

# 龙虎榜类型常量
LHB_TYPES = {
    "daily_up_7": "G0007",       # 日涨幅偏离值7%
    "daily_down_7": "G0006",    # 日跌幅偏离值7%
    "amplitude_15": "G0005",     # 日振幅15%
    "consec_3d_20": "G0008",    # 连续三日累计20%
    "turnover_20": "T0020",      # 日换手率20%
    "turnover_20_st": "T0021",
}


def fetch_lhb_summary(symbol: str, trade_date: str,
                      lookback_days: int = 3) -> dict:
    """拉取最近 lookback_days 天龙虎榜列表 + 买卖席位明细。"""
    start = dc.get_prev_trade_date(trade_date, n=lookback_days - 1)
    lhb_list = dc.get_lhb_list(start_date=start, end_date=trade_date,
                               symbol=symbol)
    if lhb_list is None or lhb_list.empty:
        return {"symbol": symbol, "on_lhb": False, "records": [],
                "buy_detail": [], "sell_detail": []}

    records = lhb_list.to_dict(orient="records")
    buy_detail = dc.get_lhb_detail(symbol=symbol, start_date=start,
                                   end_date=trade_date, side="buy")
    sell_detail = dc.get_lhb_detail(symbol=symbol, start_date=start,
                                   end_date=trade_date, side="sell")
    return {
        "symbol": symbol,
        "on_lhb": True,
        "records": records,
        "buy_detail": buy_detail.to_dict(orient="records") if buy_detail is not None and not buy_detail.empty else [],
        "sell_detail": sell_detail.to_dict(orient="records") if sell_detail is not None and not sell_detail.empty else [],
    }


def analyze_seat_quality(buy_detail: list, sell_detail: list) -> dict:
    """对席位做量化评分（用于模块一/三）。

    指标：
      net_top5_buy     : 买方前5合计金额
      net_top5_sell    : 卖方前5合计金额
      top5_net         : 前五席位净额（买-卖）
      has_institution  : 是否含"机构专用"席位
      institution_net  : 机构席位净额
      concentration    : 买方首位金额 / 买方前5金额（席位集中度，>0.3 视为游资主导）
      seat_score       : 综合席位质量分（0-100，越高越利于后续）
    """
    if not buy_detail and not sell_detail:
        return _empty_seat()

    buy_df = pd.DataFrame(buy_detail)
    sell_df = pd.DataFrame(sell_detail)

    top5_buy = float(buy_df["b_value"].sum()) if not buy_df.empty else 0.0
    top5_sell = float(sell_df["s_value"].sum()) if not sell_df.empty else 0.0
    top5_net = top5_buy - top5_sell

    has_inst = False
    inst_net = 0.0
    for df, val_col, side in [(buy_df, "b_value", "buy"),
                              (sell_df, "s_value", "sell")]:
        if df.empty or "agency" not in df.columns:
            continue
        inst_mask = df["agency"].astype(str).str.contains("机构专用")
        if inst_mask.any():
            has_inst = True
            amt = float(df.loc[inst_mask, val_col].sum())
            inst_net += amt if side == "buy" else -amt

    concentration = 0.0
    if not buy_df.empty and "b_value" in buy_df.columns and top5_buy > 0:
        concentration = float(buy_df["b_value"].iloc[0] / top5_buy)

    # 综合评分（启发式）
    score = 50.0
    if top5_net > 0:
        score += min(20.0, top5_net / 1e8 * 2.0)        # 每亿元+2分，封顶20
    if has_inst and inst_net > 0:
        score += 10.0
    if 0.3 <= concentration <= 0.5:
        score += 10.0
    if concentration > 0.5:
        score -= 5.0  # 过度集中，存在砸盘风险
    if top5_sell > top5_buy * 1.2:
        score -= 15.0  # 净卖出
    score = max(0.0, min(100.0, score))

    return {
        "net_top5_buy": top5_buy,
        "net_top5_sell": top5_sell,
        "top5_net": top5_net,
        "has_institution": has_inst,
        "institution_net": inst_net,
        "concentration": concentration,
        "seat_score": round(score, 1),
    }


def _empty_seat() -> dict:
    return {
        "net_top5_buy": 0.0, "net_top5_sell": 0.0, "top5_net": 0.0,
        "has_institution": False, "institution_net": 0.0,
        "concentration": 0.0, "seat_score": 50.0,
    }


def analyze(symbol: str, trade_date: str, lookback_days: int = 3) -> dict:
    summary = fetch_lhb_summary(symbol, trade_date, lookback_days)
    seat = analyze_seat_quality(summary.get("buy_detail", []),
                               summary.get("sell_detail", []))
    summary["seat_quality"] = seat
    return summary


def main():
    ap = argparse.ArgumentParser(description="龙虎榜分析")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--date", default=None)
    ap.add_argument("--lookback", type=int, default=3)
    args = ap.parse_args()
    trade_date = args.date or dc.get_last_trade_date("SH")
    result = analyze(args.symbol, trade_date, lookback_days=args.lookback)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
