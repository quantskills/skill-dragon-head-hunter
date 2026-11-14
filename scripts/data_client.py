"""PandaData SDK 封装层。

职责：
1. 从 .env 读取凭据（明文仅出现在 .env 中，禁止打印或写入日志）。
2. 初始化 token，仅初始化一次。
3. 暴露统一的数据获取接口，统一返回 pandas.DataFrame，并附加基本错误处理。

所有上层脚本（dragon_scorer / market_env / lhb_analyzer / concept_heat）
均通过本模块访问 PandaData，避免凭据散落。
"""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - python-dotenv 缺失时退化为 os.environ
    load_dotenv = None

try:
    import panda_data
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "未安装 panda_data，请在 skill 根目录执行: pip install -r requirements.txt"
    ) from e

# ---------------------------------------------------------------------------
# 凭据加载
# ---------------------------------------------------------------------------
_SKILL_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _SKILL_ROOT / ".env"

if load_dotenv is not None and _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

_USERNAME = os.environ.get("PANDA_USERNAME")
_PASSWORD = os.environ.get("PANDA_PASSWORD")

if not _USERNAME or not _PASSWORD:
    raise RuntimeError(
        "缺少 PandaData 凭据。请在 skill 根目录的 .env 中配置 PANDA_USERNAME / PANDA_PASSWORD"
    )

_init_lock = threading.Lock()
_initialized = False


def _ensure_init() -> None:
    """线程安全地完成 token 初始化（进程内只执行一次）。"""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        panda_data.init_token(username=_USERNAME, password=_PASSWORD)
        _initialized = True


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def _as_list(x: Union[None, str, Iterable[str]]) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def _to_df(resp) -> pd.DataFrame:
    """SDK 返回的可能是 DataFrame / list / dict / 标量，统一成 DataFrame。"""
    if resp is None:
        return pd.DataFrame()
    if isinstance(resp, pd.DataFrame):
        return resp
    if isinstance(resp, list):
        return pd.DataFrame(resp) if resp else pd.DataFrame()
    if isinstance(resp, dict):
        return pd.DataFrame([resp])
    if isinstance(resp, (str, int, float)):
        return pd.DataFrame([{"value": resp}])
    # 兜底：尝试构造，失败则返回空
    try:
        return pd.DataFrame(resp)
    except Exception:
        return pd.DataFrame()


def _to_scalar(resp, field: str = "date") -> Optional[str]:
    """从 SDK 返回中提取标量值（兼容 str / DataFrame / dict / list 形态）。"""
    if resp is None:
        return None
    if isinstance(resp, str):
        return resp
    if isinstance(resp, (int, float)):
        return str(resp)
    if isinstance(resp, dict):
        return str(resp.get(field) or next(iter(resp.values())))
    if isinstance(resp, list):
        if not resp:
            return None
        first = resp[0]
        if isinstance(first, dict):
            return str(first.get(field) or next(iter(first.values())))
        return str(first)
    if isinstance(resp, pd.DataFrame):
        if resp.empty:
            return None
        if field in resp.columns:
            return str(resp.iloc[0][field])
        return str(resp.iloc[0, 0])
    return str(resp)


# ---------------------------------------------------------------------------
# 交易日历
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def get_last_trade_date(exchange: str = "SH") -> str:
    _ensure_init()
    return _to_scalar(panda_data.get_last_trade_date(exchange=exchange), field="date")


@lru_cache(maxsize=512)
def get_prev_trade_date(date: str, exchange: str = "SH", n: int = 1) -> str:
    _ensure_init()
    return _to_scalar(
        panda_data.get_prev_trade_date(date=date, exchange=exchange, n=n),
        field="date",
    )


def get_trade_cal(start_date: str, end_date: str,
                  exchange: str = "SH", is_trading_day: Optional[int] = 1) -> pd.DataFrame:
    _ensure_init()
    return _to_df(panda_data.get_trade_cal(
        start_date=start_date, end_date=end_date,
        exchange=exchange, is_trading_day=is_trading_day, fields=[],
    ))


# ---------------------------------------------------------------------------
# 行情
# ---------------------------------------------------------------------------
def get_stock_daily(symbol: Union[str, List[str]], start_date: str, end_date: str,
                    fields: Optional[List[str]] = None, indicator: str = "",
                    st: bool = True) -> pd.DataFrame:
    """A股日线（不复权）。返回字段含 limit_up / limit_down / volume / amount。"""
    _ensure_init()
    return _to_df(panda_data.get_stock_daily(
        symbol=_as_list(symbol) or [],
        start_date=start_date, end_date=end_date,
        fields=fields or [], indicator=indicator, st=st,
    ))


def get_stock_daily_pre(symbol: Union[str, List[str]], start_date: str, end_date: str,
                        fields: Optional[List[str]] = None) -> pd.DataFrame:
    """前复权日线，用于均线计算。"""
    _ensure_init()
    return _to_df(panda_data.get_stock_daily_pre(
        symbol=_as_list(symbol) or [],
        start_date=start_date, end_date=end_date,
        fields=fields or [],
    ))


def get_stock_min(symbol: Union[str, List[str]], start_date: str, end_date: str,
                  frequency: str = "1m",
                  time_zone: Optional[tuple] = None,
                  fields: Optional[List[str]] = None) -> pd.DataFrame:
    """A股分钟线，用于识别封板时间/炸板/回封。"""
    _ensure_init()
    kwargs = dict(
        symbol=_as_list(symbol) or [],
        start_date=start_date, end_date=end_date,
        fields=fields or [], frequency=frequency,
    )
    if time_zone is not None:
        kwargs["time_zone"] = time_zone
    return _to_df(panda_data.get_stock_min(**kwargs))


def get_stock_rt_daily(symbol: Union[None, str, List[str]] = None,
                       fields: Optional[List[str]] = None) -> pd.DataFrame:
    """全市场当日日线（含所有在售股票）。用于扫描涨停家数、炸板率。

    symbol 传 None 或 [] 即可拿到全市场。
    """
    _ensure_init()
    return _to_df(panda_data.get_stock_rt_daily(
        symbol=[] if symbol is None else _as_list(symbol) or [],
        fields=fields or [],
    ))


# ---------------------------------------------------------------------------
# 龙虎榜
# ---------------------------------------------------------------------------
def get_lhb_list(start_date: str, end_date: str,
                 symbol: Union[None, str, List[str]] = None,
                 lhb_type: Optional[str] = None,
                 fields: Optional[List[str]] = None) -> pd.DataFrame:
    _ensure_init()
    return _to_df(panda_data.get_lhb_list(
        symbol=_as_list(symbol) if symbol else "",
        type=lhb_type if lhb_type else "",
        start_date=start_date, end_date=end_date,
        fields=fields or [],
    ))


def get_lhb_detail(symbol: Union[str, List[str]], start_date: str, end_date: str,
                   side: str = "buy", lhb_type: Optional[str] = None,
                   fields: Optional[List[str]] = None) -> pd.DataFrame:
    _ensure_init()
    return _to_df(panda_data.get_lhb_detail(
        symbol=_as_list(symbol) or [],
        type=lhb_type if lhb_type else "",
        start_date=start_date, end_date=end_date,
        side=side, fields=fields or [],
    ))


# ---------------------------------------------------------------------------
# 行业 / 概念
# ---------------------------------------------------------------------------
def get_stock_industry(stock_symbol: str, level: str = "L1") -> pd.DataFrame:
    _ensure_init()
    return _to_df(panda_data.get_stock_industry(stock_symbol=stock_symbol, level=level))


def get_industry_constituents(industry_code: str, level: str = "L1",
                              stock_symbol: Optional[str] = None,
                              fields: Optional[List[str]] = None) -> pd.DataFrame:
    _ensure_init()
    return _to_df(panda_data.get_industry_constituents(
        industry_code=industry_code,
        stock_symbol=stock_symbol or "",
        level=level, fields=fields or [],
    ))


def get_concept_list(concept: str, start_date: str, end_date: str) -> pd.DataFrame:
    _ensure_init()
    return _to_df(panda_data.get_concept_list(
        concept=concept, start_date=start_date, end_date=end_date,
    ))


def get_concept_constituents(concept: str, date: Optional[str] = None,
                             concept_stock: Optional[str] = None,
                             fields: Optional[List[str]] = None) -> pd.DataFrame:
    _ensure_init()
    kwargs = dict(concept=concept, concept_stock=concept_stock or "",
                  fields=fields or [])
    if date:
        kwargs["date"] = date
    return _to_df(panda_data.get_concept_constituents(**kwargs))


def get_trade_list(date: str, exchange: str = "SH") -> pd.DataFrame:
    _ensure_init()
    return _to_df(panda_data.get_trade_list(date=date, exchange=exchange))
