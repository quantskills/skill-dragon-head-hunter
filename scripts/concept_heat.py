"""板块效应分析。

输入一只股票，定位其所属申万一级行业，并扫描板块内所有成分股当日表现，
输出涨停家数、炸板数、跟风股数、板块涨幅中位数、板块成交额等指标。

这些指标直接供模块一（龙头识别）的"板块效应"维度评分使用。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import data_client as dc
from utils import is_limit_up, is_broken_limit, is_touched_limit


def locate_industry(symbol: str, level: str = "L1") -> dict | None:
    df = dc.get_stock_industry(stock_symbol=symbol, level=level)
    if df is None or df.empty:
        return None
    r = df.iloc[0]
    return {
        "industry_code": r.get("industry_code"),
        "industry_name": r.get("industry_name") or r.get("parent_l1_name"),
    }


def fetch_industry_constituents(industry_code: str, level: str = "L1") -> pd.DataFrame:
    return dc.get_industry_constituents(industry_code=industry_code, level=level)


def scan_sector_heat(industry_code: str, trade_date: str,
                     level: str = "L1") -> dict:
    """统计指定行业在某交易日的整体热度。

    指标：
      total           : 板块成分股数
      limit_up_count  : 涨停家数
      broken_count    : 炸板家数（盘中触及涨停但未封住）
      touched_count   : 盘中触及涨停家数
      follow_count   : 跟风股数（涨幅>=5%且未涨停）
      median_pct      : 板块涨幅中位数
      total_amount    : 板块当日成交额合计
      limit_up_stocks : 涨停股代码列表
    """
    cons = fetch_industry_constituents(industry_code, level=level)
    if cons is None or cons.empty:
        return _empty_sector_heat(industry_code)

    symbols = cons["stock_symbol"].dropna().unique().tolist()
    daily = dc.get_stock_daily(symbol=symbols, start_date=trade_date,
                               end_date=trade_date, fields=[])
    if daily is None or daily.empty:
        return _empty_sector_heat(industry_code)

    daily = daily[daily["date"].astype(str) == str(trade_date)].copy()
    if daily.empty:
        return _empty_sector_heat(industry_code)

    daily["pct"] = (daily["close"] / daily["pre_close"] - 1.0)
    daily = daily.dropna(subset=["pct"])

    limit_up_mask = daily.apply(is_limit_up, axis=1)
    broken_mask = daily.apply(is_broken_limit, axis=1)
    touched_mask = daily.apply(is_touched_limit, axis=1)

    limit_up_stocks = daily.loc[limit_up_mask, "symbol"].tolist()
    follow_mask = (daily["pct"] >= 0.05) & (~limit_up_mask)

    return {
        "industry_code": industry_code,
        "trade_date": str(trade_date),
        "total": int(len(daily)),
        "limit_up_count": int(limit_up_mask.sum()),
        "broken_count": int(broken_mask.sum()),
        "touched_count": int(touched_mask.sum()),
        "follow_count": int(follow_mask.sum()),
        "median_pct": float(daily["pct"].median()) if not daily.empty else 0.0,
        "total_amount": float(daily["amount"].sum()) if "amount" in daily else 0.0,
        "limit_up_stocks": limit_up_stocks,
    }


def _empty_sector_heat(industry_code: str) -> dict:
    return {
        "industry_code": industry_code, "trade_date": None,
        "total": 0, "limit_up_count": 0, "broken_count": 0, "touched_count": 0,
        "follow_count": 0, "median_pct": 0.0, "total_amount": 0.0,
        "limit_up_stocks": [],
    }


def analyze(symbol: str, trade_date: str, level: str = "L1") -> dict:
    ind = locate_industry(symbol, level=level)
    if not ind:
        return {"symbol": symbol, "industry": None, "heat": _empty_sector_heat("")}

    heat = scan_sector_heat(ind["industry_code"], trade_date, level=level)
    heat["industry_name"] = ind.get("industry_name")
    return {"symbol": symbol, "industry": ind, "heat": heat}


def main():
    ap = argparse.ArgumentParser(description="板块效应分析")
    ap.add_argument("--symbol", required=True, help="股票代码，如 000001.SZ")
    ap.add_argument("--date", default=None, help="交易日期 YYYYMMDD，默认最新交易日")
    ap.add_argument("--level", default="L1",
                    choices=["L1", "L2", "L3"], help="行业级别")
    args = ap.parse_args()

    trade_date = args.date or dc.get_last_trade_date("SH")
    result = analyze(args.symbol, trade_date, level=args.level)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
