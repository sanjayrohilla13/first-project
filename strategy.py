"""
Hourly / daily / minute Nifty backtest + grid-search optimizer
Strategy: VWAP regime anchor  +  ADX trend gate  +  RSI momentum  +  call-vs-put OI flow
Direction comes from the Nifty hourly signals. Two ways to trade it (options.mode):
  credit_spread - SELL options, always hedged with a long leg priced under hedge_max_premium
                  (Rs 10). Bullish: put credit spread; bearish: call credit spread;
                  neutral (optional): iron condor, or sell the condor always (spread.structure). Short leg chosen by spread.short_selection.
  long_option   - buy a call (bullish) / put (bearish).
Prices come from the Dhan options file.

Usage:
    python strategy.py --config configurations/config.yaml             # single run
    python strategy.py --config configurations/config.yaml --optimize  # grid search
"""

import argparse
import copy
from datetime import datetime
import itertools
import logging
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

IST = "Asia/Kolkata"


# ────────────────────────────── Helpers ──────────────────────────────
def save_csv(df: pd.DataFrame, path, **kw) -> Path:
    """Write a CSV; if the file is locked (e.g. open in Excel) save a timestamped copy instead."""
    path = Path(path)
    try:
        df.to_csv(path, **kw)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        df.to_csv(alt, **kw)
        log.warning(f"{path} is locked (open in another program?); saved to {alt} instead")
        return alt


CONFIG_DIR = Path(__file__).resolve().parent / "configurations"


def resolve_config(path: str) -> str:
    """Use the path as given if it exists, else look for it in the configurations/ folder."""
    p = Path(path)
    return str(p) if p.exists() else str(CONFIG_DIR / p.name)


def load_config(path="configurations/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"], index_col="datetime")
    df.index = (
        df.index.tz_localize("UTC").tz_convert(IST)
        if df.index.tz is None
        else df.index.tz_convert(IST)
    )
    df = df.sort_index()
    df.index.name = "datetime"
    return df


def resample(df: pd.DataFrame, timeframe: str | None) -> pd.DataFrame:
    if not timeframe:
        return df
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum"}
    if "oi" in df.columns:
        ohlc["oi"] = "last"
    return df.resample(timeframe, label="right", closed="right").agg(ohlc).dropna()


def _tf_minutes(timeframe: str) -> int:
    return int(pd.Timedelta(timeframe).total_seconds() // 60)


def resample_bars(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate finer bars (e.g. 1-minute) into `timeframe` bars. Bars stay stamped at their START
    (like the downloaded data) and buckets are anchored to the 09:15 open, so 5min gives 09:15, 09:20, ..."""
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "oi" in df.columns:
        ohlc["oi"] = "last"
    r = df.resample(timeframe, label="left", closed="left", offset="555min")  # 555 min = 09:15
    return r.agg(ohlc).dropna(subset=["open"])


def resample_options(opts: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Option bars aggregated the same way, per contract (option type + absolute strike).
    spot, strike_offset, OI and IV come from the contract's last minute in the bucket, which is what is
    known when the bar closes. If two strikes then claim the same offset (ATM moved inside the bucket,
    the older reading loses), one row per bar/type/offset is kept."""
    o = opts.sort_values("datetime")
    o["_t"] = o["datetime"]
    day0 = o["datetime"].dt.normalize()
    mins = (o["datetime"] - day0).dt.total_seconds() // 60 - 555
    step = _tf_minutes(timeframe)
    o["datetime"] = day0 + pd.to_timedelta((mins // step) * step + 555, unit="min")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
           "oi": "last", "iv": "last", "spot": "last", "strike_offset": "last", "_t": "last"}
    for c in ("expiry_flag", "expiry_code"):
        if c in o.columns:
            agg[c] = "last"
    out = o.groupby(["datetime", "option_type", "strike"], sort=False).agg(agg).reset_index()
    out = (out.sort_values("_t").drop_duplicates(["datetime", "option_type", "strike_offset"], keep="last")
           .drop(columns="_t").sort_values(["datetime", "option_type", "strike"]).reset_index(drop=True))
    return out


def load_options(path) -> pd.DataFrame:
    o = pd.read_csv(path, parse_dates=["datetime"])
    o["datetime"] = (
        o["datetime"].dt.tz_localize("UTC").dt.tz_convert(IST)
        if o["datetime"].dt.tz is None
        else o["datetime"].dt.tz_convert(IST)
    )
    return o.dropna(subset=["strike", "open", "high", "low", "close"]).reset_index(drop=True)


def compute_oi_flow(opts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-bar OI change summed over calls and puts, matched strike by strike.

    A strike's change counts only when the same strike/side exists in the
    immediately preceding bar, so strikes drifting in and out of the ATM window
    do not create fake OI jumps. Bars where total OI collapses (expiry roll to
    the next contract) are ignored.
    """
    oi_cfg = cfg["indicators"]["oi"]
    o = opts[opts["strike_offset"].abs() <= oi_cfg["strike_window"]]
    o = o.dropna(subset=["oi"]).sort_values("datetime")

    times = o["datetime"].drop_duplicates().tolist()
    prev_bar = dict(zip(times[1:], times[:-1]))
    g = o.groupby(["option_type", "strike"])
    o = o.assign(prev_dt=g["datetime"].shift(), prev_oi=g["oi"].shift())
    ok = o["prev_dt"] == o["datetime"].map(prev_bar)
    o["doi"] = (o["oi"] - o["prev_oi"]).where(ok, 0.0)

    total = o.groupby("datetime")["oi"].sum()
    rolled = (total / total.shift()) < (1 - oi_cfg["roll_drop_pct"] / 100)
    o.loc[o["datetime"].map(rolled).fillna(False).astype(bool), "doi"] = 0.0
    if rolled.any():
        log.info(f"OI flow: ignored {int(rolled.sum())} expiry-roll bars")

    flow = o.pivot_table(index="datetime", columns="option_type", values="doi",
                         aggfunc="sum").reindex(times).fillna(0.0)
    return flow.rename(columns={"CE": "call_doi", "PE": "put_doi"}).reindex(
        columns=["call_doi", "put_doi"], fill_value=0.0)


def build_option_index(opts: pd.DataFrame, roll_ratio: float = 3.0) -> dict:
    """Backtest lookups, all keyed by bar timestamp.

    bars[(ts, CE/PE, strike)]    -> (open, high, low, close)
    strikes[(ts, CE/PE, offset)] -> strike
    oi[(ts, CE/PE, strike)]      -> open interest
    max_oi[(ts, CE/PE)]          -> largest OI on that side that bar
    walls[CE/PE][ts]             -> strike with the most OI: calls at/above ATM
                                    (resistance), puts at/below ATM (support)
    step                         -> strike spacing (e.g. 50)
    expiry_day[date]             -> the date the contract traded on `date` expires

    Dhan's rolling data silently switches to next week's contract after each expiry, so a
    position held past expiry would be priced off the wrong contract. Expiry days are found
    from the ATM straddle: it decays to ~0 at expiry and jumps back up (by more than
    `roll_ratio` times) when the next contract takes over on the following day.
    """
    bars, strikes, oi = {}, {}, {}
    for ts, ot, off, k, o, h, l, c, i in zip(
        opts["datetime"], opts["option_type"], opts["strike_offset"], opts["strike"],
        opts["open"], opts["high"], opts["low"], opts["close"], opts["oi"],
    ):
        bars[(ts, ot, k)] = (o, h, l, c)
        strikes[(ts, ot, off)] = k
        if pd.notna(i):
            oi[(ts, ot, k)] = i

    has_oi = opts.dropna(subset=["oi"])
    max_oi = has_oi.groupby(["datetime", "option_type"])["oi"].max().to_dict()

    def wall(side: str, keep) -> dict:
        d = has_oi[(has_oi["option_type"] == side) & keep(has_oi["strike_offset"])]
        top = d.loc[d.groupby("datetime")["oi"].idxmax()]
        return dict(zip(top["datetime"], top["strike"]))

    walls = {"CE": wall("CE", lambda o: o >= 0), "PE": wall("PE", lambda o: o <= 0)}
    chain = {
        key: (g["strike"].to_numpy(), g["close"].to_numpy())
        for key, g in opts.groupby(["datetime", "option_type"])
    }
    first = opts[opts["datetime"] == opts["datetime"].iloc[0]]["strike"]
    step = float(first.drop_duplicates().sort_values().diff().min())
    atm = opts[opts["strike_offset"] == 0].groupby(["datetime", "option_type"])["close"].last().unstack()
    sv = (atm["CE"] + atm["PE"]).dropna()
    by_day = pd.Series(sv.index.date, index=sv.index)
    first, last = sv.groupby(by_day).first(), sv.groupby(by_day).last()
    days = list(first.index)
    jump = first / last.shift(1).replace(0, np.nan)
    rolls = [days[i - 1] for i in range(1, len(days)) if jump.iloc[i] > roll_ratio]
    expiry_day = {d: next((e for e in rolls if e >= d), days[-1]) for d in days}
    return {"bars": bars, "strikes": strikes, "oi": oi, "max_oi": max_oi,
            "walls": walls, "step": step, "chain": chain, "expiry_day": expiry_day}


def oi_entry_check(idx: dict, ec: dict, prev_ts, prev2_ts, opt_type: str, strike) -> str | None:
    """OI checks for the option we are about to buy, using only data known at the
    signal bar (prev_ts). Returns the reason for rejecting the entry, else None."""
    if not ec["enabled"]:
        return None
    oi_now = idx["oi"].get((prev_ts, opt_type, strike))
    if oi_now is None:
        return "no_oi"

    top = idx["max_oi"].get((prev_ts, opt_type))
    if top and oi_now < ec["min_rel_oi"] * top:
        return "illiquid"

    if ec["require_buildup"]:
        oi_prev = idx["oi"].get((prev2_ts, opt_type, strike))
        c_now = idx["bars"].get((prev_ts, opt_type, strike))
        c_prev = idx["bars"].get((prev2_ts, opt_type, strike))
        if oi_prev is None or c_now is None or c_prev is None:
            return "no_buildup_data"
        if not (oi_now > oi_prev and c_now[3] > c_prev[3]):
            return "no_buildup"

    atm = idx["strikes"].get((prev_ts, opt_type, 0))
    wall = idx["walls"][opt_type].get(prev_ts)
    if atm is not None and wall is not None and ec["wall_min_steps"] > 0:
        room = (wall - atm) if opt_type == "CE" else (atm - wall)
        if room < ec["wall_min_steps"] * idx["step"]:
            return "oi_wall"
    return None


# ─────────────────────────── Indicators ──────────────────────────────
def vwap(df: pd.DataFrame, anchor: str = "D") -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan).fillna(0)
    pv = tp * vol
    if anchor == "D":
        groups = df.index.normalize()
    elif anchor == "W":
        groups = df.index.strftime("%Y-%W")
    else:
        raise ValueError(f"Unsupported VWAP anchor: {anchor}")
    cum_pv = pv.groupby(groups).cumsum()
    cum_v = vol.groupby(groups).cumsum().replace(0, np.nan)
    return cum_pv / cum_v


def adx(df: pd.DataFrame, period: int = 14):
    h, l, c = df["high"], df["low"], df["close"]
    up, down = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    mdi = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean(), pdi, mdi


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def cmo(series: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator: 100 * (sum of up moves - sum of down moves) / (their total)
    over `period` bars. Ranges -100..+100; positive = buying pressure."""
    d = series.diff()
    up = d.clip(lower=0).rolling(period).sum()
    dn = (-d.clip(upper=0)).rolling(period).sum()
    return 100 * (up - dn) / (up + dn).replace(0, np.nan)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - df["close"].shift()).abs(),
         (df["low"] - df["close"].shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def supertrend(df: pd.DataFrame, period: int, multiplier: float):
    hl2 = (df["high"] + df["low"]) / 2
    a = atr(df, period)
    up, dn = hl2 + multiplier * a, hl2 - multiplier * a
    close = df["close"].to_numpy()
    up_v, dn_v = up.to_numpy().copy(), dn.to_numpy().copy()
    direction = np.ones(len(df), dtype=int)
    st = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        if close[i] > up_v[i - 1]:
            direction[i] = 1
        elif close[i] < dn_v[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            if direction[i] == 1 and dn_v[i] < dn_v[i - 1]:
                dn_v[i] = dn_v[i - 1]
            if direction[i] == -1 and up_v[i] > up_v[i - 1]:
                up_v[i] = up_v[i - 1]
        st[i] = dn_v[i] if direction[i] == 1 else up_v[i]
    return pd.Series(st, index=df.index), pd.Series(direction, index=df.index)


# ─────────────────────── Signal construction ─────────────────────────
def contrarian_score(df: pd.DataFrame, dc: dict) -> pd.Series:
    """Fade the slow trend. Positive = the market has fallen a lot recently, expect a rise.

    Averages four standardised measures of how far the market has run (prior-day move,
    5-day move, EMA20-vs-EMA50 gap, ADX-confirmed DI spread) and flips the sign. The
    features were chosen on the year BEFORE the options data (Nov 2023 - Oct 2024) and
    held up on the two later years. Each is standardised with expanding statistics
    from earlier bars only, so nothing here looks ahead.
    """
    day = df.index.normalize()
    daily_move = df.groupby(day)["close"].last() - df.groupby(day)["open"].first()
    feats = pd.DataFrame({
        "prev_day": pd.Series(day, index=df.index).map(daily_move.shift(1)) / df["atr"],
        "ret_35": df["close"].diff(35) / df["atr"],
        "ema_gap": (df["close"].ewm(span=20).mean() - df["close"].ewm(span=50).mean()) / df["atr"],
        "di_adx": (df["plus_di"] - df["minus_di"]) / 100 * (df["adx"] > dc.get("di_adx_min", 25)),
    })
    n = dc.get("min_history", 250)
    mu = feats.expanding(min_periods=n).mean().shift(1)
    sd = feats.expanding(min_periods=n).std().shift(1)
    z = ((feats - mu) / sd).clip(-3, 3)
    return -z.mean(axis=1, skipna=False)


def build_signals(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ind = cfg["indicators"]
    df = df.copy()

    df["vwap"] = vwap(df, ind["vwap"]["anchor"])
    df["adx"], df["plus_di"], df["minus_di"] = adx(df, ind["adx"]["period"])
    df["rsi"] = rsi(df["close"], ind["rsi"]["period"])
    df["atr"] = atr(df, cfg["risk"]["atr_period"])
    cm = ind.get("cmo") or {}
    df["cmo"] = cmo(df["close"], cm.get("period", 14))
    df["st"], df["st_dir"] = supertrend(
        df,
        cfg["risk"]["supertrend_period"],
        cfg["risk"]["supertrend_multiplier"],
    )

    oi_cfg = ind["oi"]
    if oi_cfg["required"] and "put_doi" in df.columns:
        n = oi_cfg["lookback"]
        put_f = df["put_doi"].rolling(n, min_periods=1).sum()
        call_f = df["call_doi"].rolling(n, min_periods=1).sum()
        denom = (put_f.abs() + call_f.abs()).replace(0, np.nan)
        df["oi_bias"] = ((put_f - call_f) / denom).fillna(0.0)
        oi_long = df["oi_bias"] > oi_cfg["bias_threshold"]
        oi_short = df["oi_bias"] < -oi_cfg["bias_threshold"]
    else:
        oi_long = pd.Series(True, index=df.index)
        oi_short = pd.Series(True, index=df.index)

    vf = cfg.get("volume_filter") or {}
    if vf.get("enabled"):
        v = df["volume"].astype(float)
        # volume is compared with the same time of day (the 09:15 bar is always busy), mean of the last 20 days
        tod = v.groupby(df.index.time).transform(lambda x: x.rolling(20, min_periods=5).mean().shift(1))
        df["rel_volume"] = (v / tod).replace([np.inf, -np.inf], np.nan)
        df["vol_surge_level"] = df["rel_volume"].rolling(int(vf.get("surge_bars", 3)), min_periods=1).mean()
        sgn = np.sign(df["close"].diff()).fillna(0)
        n_p = int(vf.get("pressure_bars", 14))
        # OBV pressure: signed volume over total volume (-1..+1); near +-1 = every bar's volume on one side
        df["obv_pressure"] = (sgn * v).rolling(n_p).sum() / v.rolling(n_p).sum()
        trig = vf.get("triggers", ["surge"])
        block = pd.Series(False, index=df.index)
        if "surge" in trig:
            block |= df["vol_surge_level"] >= vf.get("surge_ratio", 1.8)
        if "pressure" in trig:
            block |= df["obv_pressure"].abs() >= vf.get("pressure_max", 0.5)
        df["vol_block"] = block.fillna(False)

    cd = (cfg.get("spread") or {}).get("condor") or {}
    if cd:  # used by structure "adaptive"
        df["neutral_signal"] = (df["adx"] < cd["adx_max"]) & df["rsi"].between(*cd["rsi_range"])
    if "iv_atm" in df.columns:
        n_iv = cfg["spread"].get("iv_lookback", 49)
        df["iv_ratio"] = df["iv_atm"] / df["iv_atm"].rolling(n_iv, min_periods=min(10, n_iv)).mean()
    # logic "trend": every indicator in direction.trend_use must agree
    use = (cfg.get("direction") or {}).get("trend_use") or ["vwap", "adx", "di", "rsi", "oi"]
    adx_ok = df["adx"] > ind["adx"]["threshold"]
    conds = {
        "vwap": (df["close"] > df["vwap"], df["close"] < df["vwap"]),
        "adx": (adx_ok, adx_ok),
        "di": (df["plus_di"] > df["minus_di"], df["minus_di"] > df["plus_di"]),
        "rsi": (df["rsi"].between(ind["rsi"]["long_min"], ind["rsi"]["long_max"]),
                df["rsi"].between(ind["rsi"]["short_min"], ind["rsi"]["short_max"])),
        "oi": (oi_long, oi_short),
        "cmo": (df["cmo"] >= cm.get("bull", 25), df["cmo"] <= cm.get("bear", -25)),
    }
    unknown = set(use) - set(conds)
    if unknown:
        raise ValueError(f"direction.trend_use: unknown indicator(s) {sorted(unknown)}")
    df["long_signal"] = pd.concat([conds[k][0] for k in use], axis=1).all(axis=1)
    df["short_signal"] = pd.concat([conds[k][1] for k in use], axis=1).all(axis=1)

    dc = cfg.get("direction") or {}
    if dc.get("logic") == "rsi":  # RSI above rsi_bull = bullish, below rsi_bear = bearish
        df["score"] = df["rsi"]
        df["long_signal"] = df["rsi"] > dc["rsi_bull"]
        df["short_signal"] = df["rsi"] < dc["rsi_bear"]
        df["neutral_signal"] = (~df["long_signal"] & ~df["short_signal"]
                                & (df["adx"] <= ind["adx"]["threshold"]))
    elif dc.get("logic") == "cmo":  # CMO >= bull = bullish, <= bear = bearish
        df["score"] = df["cmo"]
        cmo_up, cmo_dn = df["cmo"] >= cm.get("bull", 25), df["cmo"] <= cm.get("bear", -25)
        if dc.get("cmo_use_oi"):  # the call-vs-put OI flow must agree with the CMO
            df["long_signal"], df["short_signal"] = cmo_up & oi_long, cmo_dn & oi_short
            disagree = (cmo_up | cmo_dn) & ~(df["long_signal"] | df["short_signal"])
        else:
            df["long_signal"], df["short_signal"] = cmo_up, cmo_dn
            disagree = pd.Series(False, index=df.index)
        df["neutral_signal"] = (~df["long_signal"] & ~df["short_signal"]
                                & (df["adx"] <= ind["adx"]["threshold"]))
        if dc.get("on_conflict", "neutral") == "skip":  # CMO and OI disagree -> stand aside
            df["neutral_signal"] &= ~disagree
    elif dc.get("logic") == "oi":  # open interest only: put-vs-call OI flow decides the direction
        if not (oi_cfg["required"] and "put_doi" in df.columns):
            raise ValueError('direction.logic "oi" needs indicators.oi.required: true and an options file')
        df["score"] = df["oi_bias"]
        df["long_signal"], df["short_signal"] = oi_long, oi_short
        df["neutral_signal"] = ~df["long_signal"] & ~df["short_signal"]
    elif dc.get("logic") == "contrarian":  # replaces the trend-following signals above
        df["score"] = contrarian_score(df, dc)
        thr = dc["threshold"]
        df["long_signal"] = df["score"] > thr
        df["short_signal"] = df["score"] < -thr
        df["neutral_signal"] = (df["score"].abs() <= thr) & (df["adx"] <= ind["adx"]["threshold"])

    # Filters found by studying where the direction calls failed (spread.structure adaptive/directional).
    # A vetoed call is dropped (no trade), it does not turn into a condor.
    ff = cfg.get("direction_filters") or {}
    if ff.get("enabled"):
        gap = (df["close"].ewm(span=int(ff.get("ema_fast", 20)), adjust=False).mean()
               - df["close"].ewm(span=int(ff.get("ema_slow", 50)), adjust=False).mean())
        df["atr_pct"] = df["atr"] / df["close"] * 100
        veto_long = pd.Series(False, index=df.index)
        veto_short = pd.Series(False, index=df.index)
        if ff.get("no_bullish_in_downtrend"):  # EMA fast below EMA slow: do not call "up"
            veto_long |= gap < 0
        if ff.get("no_bearish_in_uptrend"):
            veto_short |= gap > 0
        if ff.get("max_abs_oi_bias_bullish") is not None and "oi_bias" in df.columns:
            veto_long |= df["oi_bias"] > ff["max_abs_oi_bias_bullish"]  # one-sided OI flow is crowded, not bullish
        if ff.get("max_abs_oi_bias_bearish") is not None and "oi_bias" in df.columns:
            veto_short |= df["oi_bias"] < -ff["max_abs_oi_bias_bearish"]
        if ff.get("max_atr_pct") is not None:  # stormy market: no directional calls
            stormy = df["atr_pct"] > ff["max_atr_pct"]
            veto_long |= stormy
            veto_short |= stormy
        df["long_signal"] &= ~veto_long
        df["short_signal"] &= ~veto_short
    return df


# ────────────────────────────── Backtest ─────────────────────────────
def _is_daily(cfg: dict) -> bool:
    """data.bar "daily": one bar per day (data_daily/ files). Signals are read at the day's close
    and trades are opened at that same close, so the next day is the first bar checked for exits."""
    return (cfg.get("data") or {}).get("bar", "intraday") == "daily"


def _bars_per_day(cfg: dict) -> float:
    if _is_daily(cfg):
        return 1.0
    if "bars_per_day" in cfg["backtest"]:
        return float(cfg["backtest"]["bars_per_day"])
    rs = cfg.get("resample") or {}
    if rs.get("enabled"):  # 375 trading minutes a day (09:15-15:30)
        return 375 / _tf_minutes(rs["timeframe"])
    return 6.5


def _in_session(ts: pd.Timestamp, cfg: dict) -> bool:
    if _is_daily(cfg):  # daily bars are stamped at midnight; there is no intraday session
        return True
    t = ts.time()
    start = pd.Timestamp(cfg["session"]["start"]).time()
    end = pd.Timestamp(cfg["session"]["end"]).time()
    return start <= t <= end


def _last_bar_of_expiry(position: dict, df: pd.DataFrame, i: int) -> bool:
    """True on the final bar of the position's expiry day (it settles there)."""
    e, d = position["expiry"], df.index[i].date()
    return e is not None and d == e and (i + 1 == len(df) or df.index[i + 1].date() != d)


DIRECTION = {1: "Bullish", -1: "Bearish", 0: "Neutral"}


def _trade_context(position: dict, row, reason: str, prev=None) -> dict:
    """Columns that say what the trade was and where Nifty was (entry bar open, exit bar close).
    Entry indicator values are those of the signal bar; exit values are those of the exit bar,
    or of the previous bar for exits taken at the open (which only know the previous bar)."""
    ind = prev if (prev is not None and reason in ("TrendStart", "VolSurge")) else row
    return {
        "strategy": position["strategy"],
        "expected_direction": DIRECTION[position["dir"]],
        "direction_entry": position.get("score", np.nan),
        "direction_exit": ind.get("score", np.nan),
        "oi_bias_entry": position.get("oi_bias", np.nan),
        "oi_bias_exit": ind.get("oi_bias", np.nan),
        "cmo_entry": position.get("cmo", np.nan),
        "cmo_exit": ind.get("cmo", np.nan),
        "entry_spot": position["entry_spot"],
        "exit_spot": row["open"] if reason == "TrendStart" else row["close"],
    }


def fixed_lots(bt: dict) -> int | None:
    """Lots for position_sizing "fixed_qty": backtest.quantity (units, a multiple of lot_size).
    Returns None for the other sizing modes."""
    if bt["position_sizing"] != "fixed_qty":
        return None
    q, lot = int(bt["quantity"]), int(bt["lot_size"])
    if q < lot or q % lot:
        raise ValueError(f"backtest.quantity ({q}) must be a positive multiple of lot_size ({lot})")
    return q // lot


def backtest(df: pd.DataFrame, cfg: dict, opt_index):
    """Buy a call on long signals / a put on short signals; premium SL/TP."""
    bars, strikes = opt_index["bars"], opt_index["strikes"]
    daily = _is_daily(cfg)
    bt, risk, oc = cfg["backtest"], cfg["risk"], cfg["options"]
    slip = bt["slippage_bps"] / 1e4
    cost = bt["costs_bps"] / 1e4
    sl_frac = oc["sl_pct"] / 100
    lot = bt["lot_size"]

    equity = bt["initial_capital"]
    equity_curve = []
    trades = []
    rejects: dict[str, int] = {}
    position = None

    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        ts, prev_ts = df.index[i], df.index[i - 1]

        # ---- manage open position -----------------------------------
        if position is not None:
            stale = position["expiry"] is not None and ts.date() > position["expiry"]
            bar = None if stale else bars.get((ts, position["type"], position["strike"]))
            exit_price, reason = None, None
            if bar is None:
                exit_price, reason = position["last_close"], "Expiry" if stale else "NoData"
            else:
                o, h, l, c = bar
                position["last_close"] = c
                if l <= position["sl"]:
                    exit_price, reason = min(position["sl"], o), "SL"
                elif h >= position["tp"]:
                    exit_price, reason = max(position["tp"], o), "TP"
                elif _last_bar_of_expiry(position, df, i):
                    exit_price, reason = c, "Expiry"
                elif risk["trailing"] == "supertrend" and row["st_dir"] == -position["dir"]:
                    exit_price, reason = c, "ST"
                elif row["adx"] < cfg["indicators"]["adx"]["chop_threshold"]:
                    exit_price, reason = c, "RegimeDecay"
                elif cfg["session"]["exit_eod"] and not _in_session(ts, cfg):
                    exit_price, reason = c, "EOD"

            if exit_price is not None:
                net_entry = position["entry"] * (1 + slip + cost)
                net_exit = exit_price * (1 - slip - cost)
                pnl = (net_exit - net_entry) * position["qty"]
                equity += pnl
                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": ts,
                    **_trade_context(position, row, reason, prev),
                    "dir": position["dir"],
                    "option": f"{position['strike']:.0f}{position['type']}",
                    "entry": position["entry"],
                    "exit": exit_price,
                    "qty": position["qty"],
                    "option_oi": position["option_oi"],
                    "oi_bias": position["oi_bias"],
                    "pnl": pnl,
                    "r_multiple": pnl / position["risk_amount"]
                    if position["risk_amount"] > 0 else 0,
                    "reason": reason,
                })
                position = None

        # ---- entries -------------------------------------------------
        # Strike is picked from the signal bar's ATM (known then) and bought at
        # the next bar's open. Same-day only: overnight gaps make the strike stale.
        # Daily bars: signal and fill are both today's close (fill column 3); a contract expiring
        # today is never opened.
        sig, sig_ts, fc = (row, ts, 3) if daily else (prev, prev_ts, 0)
        expiring = daily and opt_index["expiry_day"].get(ts.date()) == ts.date()
        if (position is None and _in_session(ts, cfg) and (daily or prev_ts.date() == ts.date())
                and not expiring):
            opt_type = "CE" if sig["long_signal"] else "PE" if sig["short_signal"] else None
            strike = strikes.get((sig_ts, opt_type, oc["strike_offset"])) if opt_type else None
            bar = bars.get((ts, opt_type, strike)) if strike is not None else None
            if bar is not None and bar[fc] >= oc["min_premium"] and i >= 2:
                why = oi_entry_check(opt_index, oc["oi_entry"], sig_ts, df.index[i - 1 if daily else i - 2],
                                     opt_type, strike)
                if why:
                    rejects[why] = rejects.get(why, 0) + 1
                    bar = None
            if bar is not None and bar[fc] >= oc["min_premium"]:
                entry = bar[fc]
                sl_pts = entry * sl_frac
                if bt["position_sizing"] == "risk_pct":
                    lots = int(equity * bt["risk_pct"] / 100 / (sl_pts * lot))
                else:
                    lots = fixed_lots(bt) or bt["lots"]
                lots = min(max(lots, 1), int(equity / (entry * lot)))  # cash-affordable
                if lots >= 1:
                    qty = lots * lot
                    position = dict(
                        dir=1 if opt_type == "CE" else -1, type=opt_type, strike=strike,
                        strategy="Long Call" if opt_type == "CE" else "Long Put",
                        entry_spot=row["close" if daily else "open"],
                        score=sig.get("score", np.nan), cmo=sig.get("cmo", np.nan),
                        expiry=opt_index["expiry_day"].get(ts.date()),
                        entry=entry, sl=entry - sl_pts, tp=entry + risk["rr_ratio"] * sl_pts,
                        qty=qty, entry_time=ts, risk_amount=sl_pts * qty, last_close=entry,
                        option_oi=opt_index["oi"].get((sig_ts, opt_type, strike)),
                        oi_bias=sig.get("oi_bias", 0.0),
                    )

        equity_curve.append({"datetime": ts, "equity": equity})

    trades_df = pd.DataFrame(trades)
    trades_df.attrs["oi_rejects"] = rejects
    equity_df = (
        pd.DataFrame(equity_curve).set_index("datetime")
        if equity_curve else pd.DataFrame(columns=["equity"])
    )
    return trades_df, equity_df


def pick_leg(idx: dict, ts, opt_type: str, want_itm: bool, band, tol: float):
    """Strike on the ITM/OTM side of ATM whose premium is nearest the middle of
    `band`, accepted only within band +- tol. Returns the strike or None."""
    atm = idx["strikes"].get((ts, opt_type, 0))
    chain = idx["chain"].get((ts, opt_type))
    if atm is None or chain is None:
        return None
    strikes, closes = chain
    above = (opt_type == "PE") == want_itm  # ITM puts and OTM calls sit above ATM
    side = strikes > atm if above else strikes < atm
    lo, hi = band[0] - tol, band[1] + tol
    ok = side & (closes >= lo) & (closes <= hi)
    if not ok.any():
        return None
    mid = (band[0] + band[1]) / 2
    j = np.flatnonzero(ok)[np.argmin(np.abs(closes[ok] - mid))]
    return float(strikes[j])


def pick_short(idx: dict, ts, opt_type: str, sp: dict):
    """Strike to SELL on side `opt_type` (PE when bullish, CE when bearish)."""
    mode = sp["short_selection"]
    if mode in ("itm_premium", "otm_premium"):
        return pick_leg(idx, ts, opt_type, mode == "itm_premium",
                        sp["short_premium"], sp["short_tolerance"])
    if mode == "oi_wall":  # sell at the strike with the most OI on that side
        wall = idx["walls"][opt_type].get(ts)
        atm = idx["strikes"].get((ts, opt_type, 0))
        if wall is None or atm is None:
            return None
        steps = ((atm - wall) if opt_type == "PE" else (wall - atm)) / idx["step"]
        return float(wall) if steps >= sp["wall_min_steps"] else None
    raise ValueError(f"Unknown spread.short_selection: {mode}")


def pick_hedge(idx: dict, ts, opt_type: str, short_k: float, cap: float, floor: float):
    """Hedge to BUY: beyond the short strike, premium strictly under `cap`. Takes the
    priciest one under the cap, i.e. the closest strike, for the most protection."""
    chain = idx["chain"].get((ts, opt_type))
    if chain is None:
        return None
    strikes, closes = chain
    beyond = strikes < short_k if opt_type == "PE" else strikes > short_k
    ok = beyond & (closes < cap) & (closes >= floor)
    if not ok.any():
        return None
    return float(strikes[np.flatnonzero(ok)[np.argmax(closes[ok])]])


def backtest_spread(df: pd.DataFrame, cfg: dict, idx: dict):
    """Sell options, always hedged. Bullish: put credit spread. Bearish: call credit
    spread. Neutral (spread.condor.enabled): iron condor.

    The structure is tracked as P = sum(short premiums) - sum(long premiums), the cost
    to close it. We collect P0 on entry and profit when P falls. Stop and target are
    set on P as a % of the entry credit. Intrabar extremes use the worst single short
    leg's high/low (only one side of a condor can spike at a time) against the long
    legs' closes.
    """
    bars = idx["bars"]
    bt, risk, sp = cfg["backtest"], cfg["risk"], cfg["spread"]
    structure = sp.get("structure", "directional")  # directional | adaptive | condor
    s = bt["slippage_bps"] / 1e4 + bt["costs_bps"] / 1e4  # applied to each leg, each side
    lot = bt["lot_size"]
    adx_trend = cfg["indicators"]["adx"]["threshold"]
    intrabar = sp.get("intrabar", "conservative")  # extreme | conservative | close
    ff = cfg.get("direction_filters") or {}
    cooldown = bool(ff.get("enabled")) and ff.get("cooldown_after_sl")  # "same_direction" | "any" | falsy
    last_sl = None  # (date, dir) of the last stop-out
    vf = cfg.get("volume_filter") or {}
    vf_on = bool(vf.get("enabled")) and "vol_block" in df.columns
    win = sp.get("entry_window")
    daily = _is_daily(cfg)
    w_lo, w_hi = (pd.Timestamp(x).time() for x in win) if win else (None, None)

    equity = bt["initial_capital"]
    equity_curve, trades, rejects = [], [], {}
    position = None

    def reject(why: str):
        rejects[why] = rejects.get(why, 0) + 1

    for i in range(1, len(df)):
        row, prev = df.iloc[i], df.iloc[i - 1]
        ts, prev_ts = df.index[i], df.index[i - 1]

        # ---- manage open position -----------------------------------
        if position is not None:
            shorts, long_c, long_o, missing = [], 0.0, 0.0, False
            # past expiry the data shows next week's contract: never price off it
            stale = position["expiry"] is not None and ts.date() > position["expiry"]
            missing = stale
            for lg in ([] if stale else position["legs"]):
                b = bars.get((ts, lg["type"], lg["strike"]))
                lg["bar_open"] = b[0] if b else lg["last"]
                if lg["short"]:
                    if b is None:
                        missing = True
                    else:
                        shorts.append(b)
                        lg["last"] = b[3]
                else:
                    # a far-OTM hedge drops out of Dhan's ATM+-10 window as spot moves;
                    # it is cheap and moves little, so carry its last price
                    long_o += b[0] if b else lg["last"]
                    lg["last"] = b[3] if b else lg["last"]
                    long_c += lg["last"]
            if not stale:
                position["last_long"] = long_c

            exit_p = reason = None
            if missing:
                exit_p, reason = position["last_p"], "Expiry" if stale else "NoData"
            else:
                sc = sum(b[3] for b in shorts)
                p_close = sc - long_c
                p_open = sum(b[0] for b in shorts) - long_o
                # Legs' extremes do not happen at the same moment (when the market rises
                # the put falls while the call rises), so combining them is optimistic
                # for targets and pessimistic for stops. `intrabar` picks the trade-off.
                p_hi = p_close + (max(b[1] - b[3] for b in shorts) if intrabar != "close" else 0)
                p_lo = p_close - (max(b[3] - b[2] for b in shorts) if intrabar == "extreme" else 0)
                position["last_p"] = p_close
                d = position["dir"]
                if p_hi >= position["sl"]:
                    exit_p, reason = (max(position["sl"], p_open) if intrabar != "close"
                                      else p_close), "SL"
                elif p_lo <= position["tp"]:
                    exit_p, reason = (min(position["tp"], p_open) if intrabar == "extreme"
                                      else p_close), "TP"
                elif _last_bar_of_expiry(position, df, i):
                    exit_p, reason = p_close, "Expiry"  # settles with the contract
                elif (d and sp.get("exit_on_trend_change")
                      and (prev["short_signal"] if d == 1 else prev["long_signal"])
                      and position["last_p"] < position["p0"] * (1 - sp.get("trend_exit_min_gain_pct", -1e9) / 100)):
                    # the PREVIOUS bar's signal turned against the trade: exit at this bar's open
                    # (optionally only when the trade is already at least trend_exit_min_gain_pct in profit)
                    exit_p, reason = p_open, "TrendChange"
                elif d and risk["trailing"] == "supertrend" and row["st_dir"] == -d:
                    exit_p, reason = p_close, "ST"
                elif d and row["adx"] < cfg["indicators"]["adx"]["chop_threshold"]:
                    exit_p, reason = p_close, "RegimeDecay"
                elif not d and prev["adx"] > adx_trend:
                    # condor: a trend is forming. Uses the PREVIOUS bar's ADX (known at the
                    # open) and exits at this bar's open, so nothing here looks ahead.
                    exit_p, reason = p_open, "TrendStart"
                elif (vf_on and vf.get("exit_condor_on_flag") and not d and prev["vol_block"]):
                    # volume surge / one-sided flow on the PREVIOUS bar: the quiet market is over.
                    # Exit at this bar's open, so nothing here looks ahead.
                    exit_p, reason = p_open, "VolSurge"
                elif cfg["session"]["exit_eod"] and not _in_session(ts, cfg):
                    exit_p, reason = p_close, "EOD"
                elif (cfg["session"].get("overnight", "always") == "winners"
                      and (i + 1 == len(df) or df.index[i + 1].date() != ts.date())
                      and position["p0"] - p_close
                      < cfg["session"].get("winner_min_pct", 0) / 100 * position["p0"]):
                    # last bar of the day and the trade is not a winner: do not carry it
                    exit_p, reason = p_close, "EOD"

            if exit_p is not None:
                long_exit = long_c if not missing else position["last_long"]
                debit = (exit_p + long_exit) * (1 + s) - long_exit * (1 - s)  # close all legs
                pnl = (position["credit"] - debit) * position["qty"]
                equity += pnl
                # per-leg prices: bar open for exits taken at the open, else the bar close
                # (SL/TP are priced on the whole structure, so their leg prices are approximate)
                at_open = reason in ("TrendStart", "VolSurge", "TrendChange") and not missing
                legs_txt = " | ".join(
                    f"{'SELL' if lg['short'] else 'BUY'} {lg['strike']:.0f}{lg['type']} "
                    f"{lg['entry']:.2f}->{(lg.get('bar_open', lg['last']) if at_open else lg['last']):.2f}"
                    for lg in position["legs"])
                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": ts,
                    **_trade_context(position, row, reason, prev),
                    "dir": position["dir"],
                    "spread": position["label"],
                    "legs": legs_txt,
                    "entry": position["p0"],
                    "exit": exit_p,
                    "qty": position["qty"],
                    "pnl": pnl,
                    "r_multiple": pnl / position["risk_amount"]
                    if position["risk_amount"] > 0 else 0,
                    "reason": reason,
                })
                if reason == "SL":
                    last_sl = (ts.date(), position["dir"])
                position = None

        # ---- entries -------------------------------------------------
        # Legs are chosen from the signal bar's premiums (known then) and traded at
        # the next bar's open. Same-day only: overnight gaps make the strikes stale.
        # Daily bars: the signal is today's close and the legs trade at today's close (fill column 3),
        # so tomorrow is the first bar checked for exits. A contract expiring today is never opened.
        sig, sig_ts, fc = (row, ts, 3) if daily else (prev, prev_ts, 0)
        expiring = daily and idx["expiry_day"].get(ts.date()) == ts.date()
        if (position is None and _in_session(ts, cfg) and (daily or prev_ts.date() == ts.date())
                and not expiring and (win is None or w_lo <= ts.time() <= w_hi)):
            bullish, bearish = bool(sig["long_signal"]), bool(sig["short_signal"])
            if structure == "condor":
                # ignore direction and sell both sides, but not into an established trend
                # (the open condor would be closed again at the next open anyway)
                bullish = bearish = False
                neutral = bool(sig["adx"] <= adx_trend)
            else:
                neutral = (structure == "adaptive" and not bullish and not bearish
                           and bool(sig.get("neutral_signal", False)))
            if cooldown and last_sl and last_sl[0] == ts.date() and (
                    (bullish or bearish) and (cooldown == "any" or last_sl[1] == (1 if bullish else -1))):
                reject("cooldown_after_sl")
                bullish = bearish = neutral = False
            if not (bullish or bearish or neutral):
                pass
            elif vf_on and sig["vol_block"] and (
                    (neutral and vf.get("skip_condor_entry", True))
                    or ((bullish or bearish) and vf.get("skip_directional_entry", False))):
                reject("volume_flag")
            elif sp.get("iv_min_ratio") and not sig.get("iv_ratio", np.nan) >= sp["iv_min_ratio"]:
                reject("iv_low")
            else:
                legs, why = [], None
                for side in (["PE"] if bullish else ["CE"] if bearish else ["PE", "CE"]):
                    sk = pick_short(idx, sig_ts, side, sp)
                    if sk is None:
                        why = "no_short_leg"
                        break
                    hk = pick_hedge(idx, sig_ts, side, sk,
                                    sp["hedge_max_premium"], sp["hedge_min_premium"])
                    if hk is None:
                        why = "no_hedge_leg"
                        break
                    legs += [dict(type=side, strike=sk, short=True),
                             dict(type=side, strike=hk, short=False)]
                nxt = [bars.get((ts, lg["type"], lg["strike"])) for lg in legs] if not why else []
                if why:
                    reject(why)
                elif any(b is None for b in nxt):
                    reject("no_bar_next")
                else:
                    so = sum(b[fc] for lg, b in zip(legs, nxt) if lg["short"])
                    lo_ = sum(b[fc] for lg, b in zip(legs, nxt) if not lg["short"])
                    p0, credit = so - lo_, so * (1 - s) - lo_ * (1 + s)
                    if p0 <= 0 or credit <= 0 or p0 < sp["min_credit"]:
                        reject("low_credit")
                    else:
                        for lg, b in zip(legs, nxt):
                            lg["last"] = lg["entry"] = b[fc]
                        sl_pts = p0 * sp["sl_pct"] / 100
                        lots = (int(equity * bt["risk_pct"] / 100 / (sl_pts * lot))
                                if bt["position_sizing"] == "risk_pct" else fixed_lots(bt) or bt["lots"])
                        top = max(lg["strike"] for lg in legs if lg["short"])
                        margin = sp["margin_pct"] / 100 * top * lot  # per lot, rough
                        lots = min(max(lots, 1), int(equity * sp["max_margin_use_pct"] / 100 / margin))
                        if lots < 1:
                            reject("no_margin")
                        else:
                            label = " + ".join(
                                f"SELL {a['strike']:.0f}{a['type']}/BUY {h['strike']:.0f}{h['type']}"
                                for a, h in zip(legs[0::2], legs[1::2]))
                            position = dict(
                                dir=1 if bullish else -1 if bearish else 0, legs=legs,
                                strategy=("Bull Put Credit Spread" if bullish else
                                          "Bear Call Credit Spread" if bearish else "Iron Condor"),
                                entry_spot=row["close" if daily else "open"],
                                score=sig.get("score", np.nan), cmo=sig.get("cmo", np.nan),
                                oi_bias=sig.get("oi_bias", np.nan),
                                expiry=idx["expiry_day"].get(ts.date()),
                                label=label, p0=p0, credit=credit, qty=lots * lot,
                                entry_time=ts, sl=p0 + sl_pts, tp=p0 * (1 - sp["tp_pct"] / 100),
                                risk_amount=sl_pts * lots * lot, last_p=p0,
                                last_long=lo_,
                            )

        # equity includes the open position's unrealised P&L, so multi-day holds show their drawdown
        unrealised = (position["p0"] - position["last_p"]) * position["qty"] if position else 0.0
        equity_curve.append({"datetime": ts, "equity": equity + unrealised})

    trades_df = pd.DataFrame(trades)
    trades_df.attrs["oi_rejects"] = rejects
    equity_df = (
        pd.DataFrame(equity_curve).set_index("datetime")
        if equity_curve else pd.DataFrame(columns=["equity"])
    )
    return trades_df, equity_df


def run_backtest(df, cfg, idx):
    if cfg["options"]["mode"] == "credit_spread":
        return backtest_spread(df, cfg, idx)
    return backtest(df, cfg, idx)


# ────────────────────────────── Metrics ──────────────────────────────
def compute_metrics(trades_df, equity_df, initial_capital, bars_per_day=6.5):
    if trades_df.empty:
        return {"total_trades": 0, "total_pnl": 0, "avg_r": 0}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    eq = equity_df["equity"]
    peak = eq.cummax()
    dd = ((eq - peak) / peak).min()
    rets = eq.pct_change().dropna()
    # annualise per-bar returns: 252 trading days x bars per day (6.5 hourly, 1 daily)
    ann_factor = np.sqrt(252 * bars_per_day)
    sharpe = (rets.mean() / rets.std()) * ann_factor if rets.std() > 0 else 0.0
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 86400) if len(eq) > 1 else 0.0
    cagr = ((eq.iloc[-1] / initial_capital) ** (1 / years) - 1) * 100 if years > 0 and eq.iloc[-1] > 0 else 0.0
    return {
        "total_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df),
        "cagr_pct": cagr,
        "total_pnl": trades_df["pnl"].sum(),
        "return_pct": trades_df["pnl"].sum() / initial_capital * 100,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
        "avg_r": trades_df["r_multiple"].mean(),
        "max_dd_pct": dd * 100,
        "sharpe": sharpe,
        "final_equity": eq.iloc[-1],
    }


# ───────────────────────────── Optimizer ─────────────────────────────
def apply_params(cfg: dict, params: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    mapping = {
        "adx_period":        ("indicators", "adx", "period"),
        "adx_threshold":     ("indicators", "adx", "threshold"),
        "rsi_period":        ("indicators", "rsi", "period"),
        "rsi_long_min":      ("indicators", "rsi", "long_min"),
        "rsi_long_max":      ("indicators", "rsi", "long_max"),
        "rsi_short_min":     ("indicators", "rsi", "short_min"),
        "rsi_short_max":     ("indicators", "rsi", "short_max"),
        "oi_bias_threshold": ("indicators", "oi", "bias_threshold"),
        "oi_lookback":       ("indicators", "oi", "lookback"),
        "sl_pct":            ("spread" if cfg["options"]["mode"] == "credit_spread"
                              else "options", "sl_pct"),
        "tp_pct":            ("spread", "tp_pct"),
        "dir_threshold":     ("direction", "threshold"),
        "oi_min_rel":        ("options", "oi_entry", "min_rel_oi"),
        "oi_wall_steps":     ("options", "oi_entry", "wall_min_steps"),
        "rr_ratio":          ("risk", "rr_ratio"),
    }
    for key, val in params.items():
        if key not in mapping:
            raise KeyError(f"No config mapping for optimize key: {key}")
        *parents, leaf = mapping[key]
        node = cfg
        for p in parents:
            node = node[p]
        node[leaf] = val
    return cfg


def grid_search(df, cfg, opt_index):
    grid = cfg["optimize"]["params"]
    keys = list(grid.keys())
    rows = []
    total = int(np.prod([len(grid[k]) for k in keys]))
    log.info(f"Grid search: {total} combinations")

    for n, combo in enumerate(itertools.product(*[grid[k] for k in keys]), 1):
        params = dict(zip(keys, combo))
        trial_cfg = apply_params(cfg, params)
        sig = build_signals(df, trial_cfg)
        trades, eq = run_backtest(sig, trial_cfg, opt_index)
        m = compute_metrics(trades, eq, trial_cfg["backtest"]["initial_capital"], _bars_per_day(trial_cfg))
        m.update(params)
        rows.append(m)
        if n % 10 == 0 or n == total:
            log.info(f"  {n}/{total} done")

    res = pd.DataFrame(rows).sort_values(
        cfg["optimize"]["sort_by"], ascending=False
    ).reset_index(drop=True)
    out = Path(cfg["optimize"]["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_csv(res, out, index=False)
    log.info(f"Optimization results -> {out}")
    log.info(f"\nTop 5:\n{res.head(5).to_string(index=False)}")
    return res


# ──────────────────────────────── Main ───────────────────────────────
# ────────────────────────────── HTML report ──────────────────────────────
def _svg_line(xs, ys, w=900, h=240, fill_to=None, color="var(--accent)", ylabel=lambda v: f"{v:,.0f}"):
    """Tiny dependency-free line/area chart. xs are Timestamps, ys floats."""
    if len(xs) < 2:
        return ""
    step = max(1, len(xs) // 700)
    xs, ys = list(xs)[::step], np.asarray(ys, float)[::step]
    padl, padr, padt, padb = 70, 12, 10, 24
    t0, t1 = xs[0].value, xs[-1].value
    lo = float(min(ys.min(), fill_to if fill_to is not None else ys.min()))
    hi = float(max(ys.max(), fill_to if fill_to is not None else ys.max()))
    if hi == lo:
        hi = lo + 1
    X = lambda t: padl + (t.value - t0) / max(t1 - t0, 1) * (w - padl - padr)
    Y = lambda v: padt + (hi - v) / (hi - lo) * (h - padt - padb)
    pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in zip(xs, ys))
    base = Y(fill_to) if fill_to is not None else h - padb
    area = f'<polygon points="{X(xs[0]):.1f},{base:.1f} {pts} {X(xs[-1]):.1f},{base:.1f}" fill="{color}" opacity=".15"/>'
    grid = ""
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        grid += (f'<line x1="{padl}" x2="{w - padr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                 f'<text x="{padl - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="ax">{ylabel(v)}</text>')
    for y in sorted({t.year for t in xs}):
        tt = pd.Timestamp(year=y, month=1, day=1, tz=xs[0].tz)
        if xs[0] <= tt <= xs[-1]:
            grid += (f'<line x1="{X(tt):.1f}" x2="{X(tt):.1f}" y1="{padt}" y2="{h - padb}" class="grid"/>'
                     f'<text x="{X(tt) + 3:.1f}" y="{h - 8}" class="ax">{y}</text>')
    return (f'<svg viewBox="0 0 {w} {h}" class="chart" role="img">{grid}{area}'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/></svg>')


def _svg_price(price: pd.DataFrame, spans: list, trades: pd.DataFrame, w=900, h=360) -> str:
    """Nifty close with EMA lines (computed on every bar) and a marker at each trade entry."""
    close = price["close"].astype(float)
    close.index = pd.to_datetime(close.index, utc=True).tz_convert("Asia/Kolkata")
    colors = ["#f59e0b", "#a855f7", "#14b8a6", "#ec4899"]
    series = [("Nifty close", close, "var(--accent)", 1.2)]
    for i, n in enumerate(spans):
        series.append((f"EMA {n}", close.ewm(span=int(n), adjust=False).mean(), colors[i % len(colors)], 1.3))
    step = max(1, len(close) // 1200)
    padl, padr, padt, padb = 70, 12, 10, 24
    t0, t1 = close.index[0].value, close.index[-1].value
    lo, hi = float(close.min()), float(close.max())
    X = lambda t: padl + (t.value - t0) / max(t1 - t0, 1) * (w - padl - padr)
    Y = lambda v: padt + (hi - v) / (hi - lo) * (h - padt - padb)
    out = ""
    for k in range(6):
        v = lo + (hi - lo) * k / 5
        out += (f'<line x1="{padl}" x2="{w - padr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                f'<text x="{padl - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="ax">{v:,.2f}</text>')
    for y in sorted({t.year for t in close.index}):
        tt = pd.Timestamp(year=y, month=1, day=1, tz="Asia/Kolkata")
        if close.index[0] <= tt <= close.index[-1]:
            out += (f'<line x1="{X(tt):.1f}" x2="{X(tt):.1f}" y1="{padt}" y2="{h - padb}" class="grid"/>'
                    f'<text x="{X(tt) + 3:.1f}" y="{h - 8}" class="ax">{y}</text>')
    for name, ser, col, sw in series:
        pts = " ".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in list(ser.items())[::step])
        out += f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}"><title>{name}</title></polyline>'
    mk = {"Bullish": ("#16a34a", -1), "Bearish": ("#dc2626", 1), "Neutral": ("#94a3b8", 0)}
    for r in trades.itertuples():
        col, _ = mk.get(r.expected_direction, mk["Neutral"])
        out += (f'<circle cx="{X(r.entry_time):.1f}" cy="{Y(float(r.entry_spot)):.1f}" r="2.2" fill="{col}" opacity=".75">'
                f'<title>{r.entry_time:%Y-%m-%d %H:%M} {r.strategy} @ {float(r.entry_spot):,.0f}</title></circle>')
    legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for n, _, c, _ in series)
    legend += "".join(f'<span><i class="dot" style="background:{c}"></i>{k} entry</span>' for k, (c, _) in mk.items())
    return f'<div class="legend">{legend}</div><svg viewBox="0 0 {w} {h}" class="chart" role="img">{out}</svg>'


def _group_table(trades: pd.DataFrame, key: str, title: str, esc) -> str:
    rows = ""
    for k, g in trades.groupby(key):
        gp, gl = g.loc[g.pnl > 0, "pnl"].sum(), -g.loc[g.pnl <= 0, "pnl"].sum()
        pf = f"{gp / gl:.2f}" if gl > 0 else "∞"
        rows += (f"<tr><td>{esc(str(k))}</td><td>{len(g)}</td><td>{(g.pnl > 0).mean():.0%}</td>"
                 f'<td class="{"pos" if g.pnl.sum() >= 0 else "neg"}">{g.pnl.sum():,.2f}</td><td>{pf}</td>'
                 f"<td>{g.pnl.mean():,.2f}</td></tr>")
    return (f"<h3>{esc(title)}</h3><table><thead><tr><th>{esc(key)}</th><th>Trades</th><th>Win</th>"
            f"<th>P&amp;L ₹</th><th>PF</th><th>Avg ₹</th></tr></thead><tbody>{rows}</tbody></table>")


def write_html_report(trades: pd.DataFrame, eq: pd.DataFrame, metrics: dict, cfg: dict,
                      config_path: str, data_range, path, price=None) -> Path:
    from html import escape as esc
    cap = cfg["backtest"]["initial_capital"]
    tz = "Asia/Kolkata"
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True).dt.tz_convert(tz)
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True).dt.tz_convert(tz)
    t["year"] = t["exit_time"].dt.year
    e = eq.copy()
    e.index = pd.to_datetime(e.index, utc=True).tz_convert(tz)
    dd = (e["equity"] / e["equity"].cummax() - 1) * 100

    def card(label, val, cls=""):
        return f'<div class="card"><div class="lab">{esc(label)}</div><div class="val {cls}">{esc(str(val))}</div></div>'

    m = metrics
    cards = "".join([
        card("Total return", f"{m['return_pct']:.1f}%", "pos" if m["return_pct"] >= 0 else "neg"),
        card("CAGR", f"{m['cagr_pct']:.1f}%", "pos" if m["cagr_pct"] >= 0 else "neg"),
        card("Net P&L", f"₹{m['total_pnl']:,.2f}", "pos" if m["total_pnl"] >= 0 else "neg"),
        card("Trades", m["total_trades"]),
        card("Win rate", f"{m['win_rate']:.1%}"),
        card("Profit factor", f"{m['profit_factor']:.2f}"),
        card("Max drawdown", f"{m['max_dd_pct']:.1f}%", "neg"),
        card("Sharpe (annualised)", f"{m['sharpe']:.2f}"),
        card("Avg R", f"{m['avg_r']:.3f}"),
        card("Final equity", f"₹{m['final_equity']:,.2f}"),
    ])

    # per-year table + monthly heatmap (% of starting capital)
    yrows = ""
    for y in sorted(t["year"].unique()):
        g = t[t.year == y]
        gp, gl = g.loc[g.pnl > 0, "pnl"].sum(), -g.loc[g.pnl <= 0, "pnl"].sum()
        pf = f"{gp / gl:.2f}" if gl > 0 else "∞"
        yrows += (f"<tr><td>{y}</td><td>{len(g)}</td><td>{(g.pnl > 0).mean():.0%}</td>"
                  f'<td class="{"pos" if g.pnl.sum() >= 0 else "neg"}">{g.pnl.sum():,.2f}</td>'
                  f"<td>{g.pnl.sum() / cap * 100:.1f}%</td><td>{pf}</td></tr>")
    mp = t.assign(mon=t["exit_time"].dt.month).pivot_table(index="year", columns="mon", values="pnl", aggfunc="sum")
    mp = mp.reindex(columns=range(1, 13)) / cap * 100
    lim = max(float(np.nanmax(np.abs(mp.values))), 0.01)
    heat = "<tr><th></th>" + "".join(f"<th>{n}</th>" for n in "J F M A M J J A S O N D".split()) + "<th>Year</th></tr>"
    for y, r in mp.iterrows():
        cells = ""
        for v in r.values:
            if np.isnan(v):
                cells += "<td></td>"
            else:
                a = min(abs(v) / lim, 1) * 0.75
                col = f"rgba(34,160,90,{a:.2f})" if v >= 0 else f"rgba(220,70,70,{a:.2f})"
                cells += f'<td style="background:{col}">{v:.1f}</td>'
        heat += f"<tr><th>{y}</th>{cells}<td><b>{np.nansum(r.values):.1f}</b></td></tr>"

    show = t.sort_values("entry_time", ascending=False)
    cols = ["entry_time", "exit_time", "strategy", "expected_direction",
            "direction_entry", "direction_exit", "oi_bias_entry", "oi_bias_exit", "cmo_entry", "cmo_exit",
            "entry_spot", "exit_spot", "spread", "legs", "entry", "exit", "qty", "pnl", "r_multiple", "reason"]
    th = "".join(f"<th>{c}</th>" for c in cols)

    def cell(c, v):
        if c == "pnl":
            return f'<td class="{"pos" if v >= 0 else "neg"}">{v:,.2f}</td>'
        if isinstance(v, pd.Timestamp):
            return f"<td>{v:%Y-%m-%d %H:%M}</td>"
        if isinstance(v, float):
            return f"<td>{v:.2f}</td>"
        return f"<td>{esc(str(v))}</td>"

    body = "".join("<tr>" + "".join(cell(c, v) for c, v in zip(cols, r)) + "</tr>"
                   for r in show[cols].itertuples(index=False))

    sp, ses, dr = cfg["spread"], cfg["session"], cfg["direction"]
    settings = {
        "Config file": config_path,
        "Data": f"{data_range[0]:%Y-%m-%d} → {data_range[1]:%Y-%m-%d}",
        "Direction logic": f"{dr['logic']} (CMO ±{(cfg['indicators'].get('cmo') or {}).get('bull', 25)}, OI agree: {dr.get('cmo_use_oi')}, "
                           f"on conflict: {dr.get('on_conflict')})",
        "Structure": sp["structure"],
        "Short leg": f"{sp['short_selection']} premium {sp['short_premium']} ±{sp['short_tolerance']}",
        "Hedge": f"premium ₹{sp['hedge_min_premium']}–{sp['hedge_max_premium']}",
        "Stop / target": f"{sp['sl_pct']}% / {sp['tp_pct']}% of credit ({sp['intrabar']} intrabar)",
        "Overnight": "flat at close" if ses["exit_eod"] else f"{ses['overnight']} (min {ses.get('winner_min_pct', 0)}% earned)",
        "Capital / sizing": (f"₹{cap:,.0f}, fixed {cfg['backtest']['quantity']} units per trade"
                             if cfg["backtest"]["position_sizing"] == "fixed_qty" else
                             f"₹{cap:,.0f}, {cfg['backtest']['position_sizing']} {cfg['backtest'].get('risk_pct')}%"),
        "Costs": f"slippage {cfg['backtest']['slippage_bps']} bps + costs {cfg['backtest']['costs_bps']} bps",
    }
    sett = "".join(f"<tr><th>{esc(k)}</th><td>{esc(str(v))}</td></tr>" for k, v in settings.items())

    price_html = ""
    if price is not None and len(price):
        spans = (cfg.get("report") or {}).get("ema", [10, 20])
        price_html = (f"<h2>Nifty with EMA {' / '.join(str(x) for x in spans)} and trade entries</h2>"
                      + _svg_price(price, spans, t))

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Backtest Report</title>
<style>
:root{{--bg:#f6f7f9;--panel:#fff;--text:#1b1f24;--mute:#667085;--line:#e3e6eb;--accent:#2563eb;--pos:#15803d;--neg:#b91c1c}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0f1318;--panel:#171c23;--text:#e6e9ee;--mute:#98a2b3;--line:#2a313b;--accent:#6ea0ff;--pos:#4ade80;--neg:#f87171}}}}
body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,Segoe UI,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:20px 16px 60px}}
h1{{margin:0 0 2px;font-size:22px}} h2{{margin:28px 0 10px;font-size:17px}} h3{{margin:18px 0 6px;font-size:14px;color:var(--mute)}}
.sub{{color:var(--mute);margin-bottom:16px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}}
.lab{{color:var(--mute);font-size:12px}} .val{{font-size:20px;font-weight:600}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}}
table{{border-collapse:collapse;width:100%;background:var(--panel);border:1px solid var(--line)}}
th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} thead th{{color:var(--mute);font-weight:600}}
.scroll{{overflow:auto;max-height:560px;border:1px solid var(--line);border-radius:8px}}
.scroll table{{border:0}} .scroll thead th{{position:sticky;top:0;background:var(--panel)}}
.heat td,.heat th{{text-align:center;padding:5px 4px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}}
.chart{{width:100%;height:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px}}
.grid{{stroke:var(--line);stroke-width:1}} .ax{{fill:var(--mute);font-size:11px}}
.note{{color:var(--mute);font-size:12px}}
.legend{{display:flex;flex-wrap:wrap;gap:6px 16px;margin-bottom:6px;font-size:12px;color:var(--mute)}}
.legend i{{display:inline-block;width:14px;height:3px;margin-right:5px;vertical-align:middle}}
.legend i.dot{{width:8px;height:8px;border-radius:50%}}
</style></head><body><main>
<h1>Nifty options backtest</h1>
<div class="sub">{esc(Path(config_path).name)} · generated {datetime.now():%Y-%m-%d %H:%M} ·
{data_range[0]:%d %b %Y} → {data_range[1]:%d %b %Y}</div>
<div class="cards">{cards}</div>
{price_html}<h2>Equity curve (mark-to-market)</h2>{_svg_line(e.index, e['equity'])}
<h2>Drawdown (%)</h2>{_svg_line(e.index, dd, h=160, fill_to=0, color="var(--neg)", ylabel=lambda v: f"{v:.1f}%")}
<h2>By year</h2>
<table><thead><tr><th>Year</th><th>Trades</th><th>Win</th><th>P&amp;L ₹</th><th>% of capital</th><th>PF</th></tr></thead><tbody>{yrows}</tbody></table>
<h2>Monthly P&amp;L (% of starting capital, by exit month)</h2><table class="heat">{heat}</table>
<h2>Breakdown</h2>
<div class="grid2"><div>{_group_table(t, 'strategy', 'By strategy', esc)}</div>
<div>{_group_table(t, 'expected_direction', 'By expected direction', esc)}</div>
<div>{_group_table(t, 'reason', 'By exit reason', esc)}</div></div>
<h2>Settings</h2><table>{sett}</table>
<h2>All trades ({len(show)}, newest first)</h2><div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>
<p class="note">Backtest results are not a forecast. Entries and exits use the configured bars (hourly or daily) with the configured intrabar assumption.</p>
</main></body></html>"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(doc, encoding="utf-8")
    except PermissionError:
        path = path.with_name(f"{path.stem}_{datetime.now():%Y%m%d_%H%M%S}{path.suffix}")
        path.write_text(doc, encoding="utf-8")
    return path


def main(config_path: str, run_opt: bool, open_report: bool = True) -> None:
    cfg = load_config(config_path)
    df = load_data(cfg["data"]["output_csv"])

    opt_path = Path(cfg["dhan"]["output_csv"])
    if not opt_path.exists():
        raise SystemExit(f"{opt_path} not found. Run: python download_data.py --only options")
    opts = load_options(opt_path)
    if cfg["resample"]["enabled"]:  # e.g. 1-minute files traded on 5-minute bars (index and options alike)
        tf = cfg["resample"]["timeframe"]
        n_min, n_opt = len(df), len(opts)
        df, opts = resample_bars(df, tf), resample_options(opts, tf)
        log.info(f"Resampled to {tf} bars: index {n_min} -> {len(df)} bars, options {n_opt} -> {len(opts)} rows "
                 f"({_bars_per_day(cfg):.0f} bars/day)")
    if "expiry_code" in opts.columns:  # the daily file can hold several expiries; the lookups need one contract per bar
        code = int(cfg["dhan"]["expiry_code"])
        opts = opts[opts["expiry_code"] == code].reset_index(drop=True)
        log.info(f"Trading expiry_code {code} (dhan.expiry_code)")
    flow = compute_oi_flow(opts, cfg)
    df = df.join(flow, how="left")
    df[["call_doi", "put_doi"]] = df[["call_doi", "put_doi"]].fillna(0.0)
    df = df.join(opts[opts["strike_offset"] == 0].groupby("datetime")["iv"].mean().rename("iv_atm"))
    opt_index = build_option_index(opts, cfg["session"].get("expiry_roll_ratio", 3.0))
    log.info(f"Expiry days detected from the option data: {len(set(opt_index['expiry_day'].values()))}")
    log.info(f"Options: {len(opts)} rows, {opts['datetime'].min():%Y-%m-%d} -> "
             f"{opts['datetime'].max():%Y-%m-%d}")

    if run_opt and cfg["optimize"]["enabled"]:
        grid_search(df, cfg, opt_index)
        return

    sig = build_signals(df, cfg)
    trades, eq = run_backtest(sig, cfg, opt_index)
    metrics = compute_metrics(trades, eq, cfg["backtest"]["initial_capital"], _bars_per_day(cfg))
    log.info(f"Signals skipped at entry: {trades.attrs.get('oi_rejects') or 'none'}")

    # results are shown to 2 decimal places everywhere (the report reads the rounded values too)
    metrics = {k: round(float(v), 2) if isinstance(v, (float, np.floating)) else v for k, v in metrics.items()}
    trades = trades.round(2)
    eq = eq.round(2)

    out_dir = Path(cfg.get("results_dir", "results"))  # a separate folder keeps daily and hourly runs apart
    out_dir.mkdir(parents=True, exist_ok=True)
    save_csv(trades, out_dir / "trades.csv", index=False)
    save_csv(eq, out_dir / "equity_curve.csv")

    log.info("── BACKTEST SUMMARY ──")
    for k, v in metrics.items():
        log.info(f"  {k:>15}: {v:.2f}" if isinstance(v, float) else f"  {k:>15}: {v}")

    if not trades.empty:
        rep = cfg.get("report", {})
        rp = write_html_report(trades, eq, metrics, cfg, config_path,
                               (df.index.min(), df.index.max()), rep.get("html", str(out_dir / "report.html")),
                               price=df)
        log.info(f"HTML report: {rp.resolve()}")
        if open_report and rep.get("open", True):
            webbrowser.open(rp.resolve().as_uri())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configurations/config.yaml")
    p.add_argument("--optimize", action="store_true")
    p.add_argument("--no-open", action="store_true", help="write results/report.html but do not open it")
    args = p.parse_args()
    main(resolve_config(args.config), args.optimize, not args.no_open)