"""
Downloads hourly Nifty data for backtesting.

Sources (config: data.source):
  dhan     - Nifty index OHLC from Dhan's intraday charts (up to 5 years, 90 days a request),
             with NIFTYBEES volume as the VWAP weight. Needs the dhan_underlying section.
  zerodha  - Nifty futures OHLCV + Open Interest from the Kite Connect
             historical API (needs a Kite Connect subscription).
  yfinance - ^NSEI prices with volume from a proxy symbol; no OI.

Nifty OPTIONS (calls and puts, every strike Dhan offers around ATM) with OHLC,
volume, IV, spot and Open Interest come from the DhanHQ rolling-option API when
dhan.enabled is true. They are written in long format to dhan.output_csv, one
row per (bar, option type, strike offset). The strategy needs both files: the
underlying (data.output_csv) for indicators, the options for OI and trading.

Every source downloads up to the run time, keeping only finished bars (the candle still
forming is dropped). Today's options are refetched on each run until the session is over.

If the Dhan token is missing or expired, Kite Connect takes over (dhan.kite_fallback: false
turns this off). The underlying is fully replaced: Nifty index + NIFTYBEES volume, the same
series Dhan gives. Options are only topped up, because Kite does not serve expired contracts:
just the days after the saved Dhan data that belong to the live weekly expiry (no IV). Those
rows live in data/kite_parts, separate from dhan_parts, and Dhan's replace them once it works.

Credentials come from environment variables (never from config.yaml):
    KITE_API_KEY, KITE_API_SECRET, KITE_ACCESS_TOKEN (optional, see --login)
    DHAN_ACCESS_TOKEN, DHAN_CLIENT_ID (optional)

Usage:
    python download_data.py --login                  # once per day: get a Kite token
    python download_data.py --config config.yaml     # underlying + options
    python download_data.py --only options           # just the Dhan options
    python download_data.py --probe                  # debug Dhan request parameters
    python download_data.py --assemble               # build options CSV from saved parts
    python download_data.py --config config_download_5y_daily.yaml --only daily   # daily candles

Daily Nifty candles (--only daily) are set up in the config's `daily` section (years, source,
output_csv) and go to their own folder, apart from the hourly data.
"""

import argparse
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

IST = "Asia/Kolkata"
TOKEN_CACHE = Path(".kite_token.json")  # access tokens expire daily (~6 AM IST)


SESSION_CLOSE = dtime(15, 30)
DAY_DONE = dtime(16, 0)  # after this, today's bars are final and today counts as fully downloaded


def _localize_ist(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC").tz_convert(IST)
    return idx.tz_convert(IST)


def _now_ist() -> datetime:
    """Run time on the exchange clock (naive, as Kite expects), whatever the PC's timezone."""
    return datetime.now(ZoneInfo(IST)).replace(tzinfo=None)


def _closed(times: pd.Series, minutes: int) -> pd.Series:
    """True for bars that have finished: the bar still forming at run time is dropped, so a
    half-built candle is never saved as if it were final. The last bar ends at the close."""
    close = times.dt.normalize() + pd.Timedelta(hours=SESSION_CLOSE.hour, minutes=SESSION_CLOSE.minute)
    bar_end = times + pd.Timedelta(minutes=minutes)
    return bar_end.where(bar_end < close, close) <= pd.Timestamp(_now_ist()).tz_localize(IST)


def _bar_minutes(cfg: dict, source: str) -> int | None:
    if source == "dhan":
        return int(cfg["dhan_underlying"]["interval"])
    if source == "zerodha":
        return {v: k for k, v in KITE_INTERVALS.items()}.get(cfg["zerodha"]["interval"])
    try:
        return int(pd.Timedelta(cfg["data"]["interval"].replace("m", "min")).total_seconds() // 60)
    except (KeyError, ValueError):
        return None


# ─────────────────────────────── Zerodha ────────────────────────────────
KITE_KEYS_HELP = """Set them with these PowerShell commands, using the values from https://developers.kite.trade/apps

  # Saved for new terminals:
  setx KITE_API_KEY "your_api_key"
  setx KITE_API_SECRET "your_api_secret"

  # setx does not change the terminal it runs in, so also run these here:
  $env:KITE_API_KEY = "your_api_key"
  $env:KITE_API_SECRET = "your_api_secret"

Run the same commands again to update them, then: python download_data.py --login"""


def _kite_client():
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import TokenException

    api_key = os.environ.get("KITE_API_KEY")
    if not api_key:
        raise SystemExit(f"KITE_API_KEY is not set.\n{KITE_KEYS_HELP}")
    kite = KiteConnect(api_key=api_key)

    token = os.environ.get("KITE_ACCESS_TOKEN")
    if not token and TOKEN_CACHE.exists():
        cached = json.loads(TOKEN_CACHE.read_text())
        if cached.get("date") == datetime.now().strftime("%Y-%m-%d"):
            token = cached["access_token"]
    if not token:
        raise SystemExit(
            "No Kite access token for today. Run: python download_data.py --login"
        )
    kite.set_access_token(token)
    try:
        kite.profile()  # fail now, with a clear message, if the token has expired
    except TokenException as e:
        raise SystemExit(f"Kite rejected the access token ({e}). "
                         "Run: python download_data.py --login")
    return kite


def kite_login() -> None:
    """Interactive login: exchange the request_token for a daily access token."""
    from kiteconnect import KiteConnect

    api_key, secret = os.environ.get("KITE_API_KEY"), os.environ.get("KITE_API_SECRET")
    if not (api_key and secret):
        missing = [n for n, v in (("KITE_API_KEY", api_key), ("KITE_API_SECRET", secret)) if not v]
        raise SystemExit(f"{' and '.join(missing)} not set.\n{KITE_KEYS_HELP}")
    kite = KiteConnect(api_key=api_key)
    print(f"1. Open and log in:\n   {kite.login_url()}")
    print("2. After login you are redirected to a URL containing ?request_token=XXXX")
    request_token = input("3. Paste the request_token here: ").strip()
    session = kite.generate_session(request_token, api_secret=secret)
    TOKEN_CACHE.write_text(json.dumps({
        "date": datetime.now().strftime("%Y-%m-%d"),
        "access_token": session["access_token"],
    }))
    log.info(f"Access token saved to {TOKEN_CACHE} (valid for today only).")


def _resolve_future_token(kite, z: dict) -> int:
    """Instrument token of the nearest-expiry (front-month) futures contract."""
    if z.get("instrument_token"):
        return int(z["instrument_token"])
    inst = pd.DataFrame(kite.instruments(z.get("exchange", "NFO")))
    futs = inst[(inst["name"] == z["underlying"]) & (inst["instrument_type"] == "FUT")]
    futs = futs[pd.to_datetime(futs["expiry"]) >= pd.Timestamp.today().normalize()]
    if futs.empty:
        raise RuntimeError(f"No active {z['underlying']} futures found on NFO.")
    row = futs.sort_values("expiry").iloc[0]
    log.info(f"Using {row['tradingsymbol']} (token {row['instrument_token']})")
    return int(row["instrument_token"])


def _kite_history(kite, token: int, start: datetime, end: datetime, interval: str,
                  chunk_days: int, label: str = "", **kwargs) -> pd.DataFrame:
    """Candles split into API-sized requests; an empty frame when Kite has none."""
    chunk = timedelta(days=chunk_days)
    frames, cur = [], start
    while cur < end:
        nxt = min(cur + chunk, end)
        log.info(f"Kite {label} {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d} | {interval}")
        candles = kite.historical_data(token, cur, nxt, interval, **kwargs)
        if candles:
            frames.append(pd.DataFrame(candles))
        cur = nxt
        time.sleep(0.4)  # stay under the 3 requests/sec limit
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).rename(columns={"date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df.index = _localize_ist(df.index)
    return df[~df.index.duplicated(keep="last")].sort_index()


def download_zerodha(cfg: dict) -> pd.DataFrame:
    d, z = cfg["data"], cfg["zerodha"]
    kite = _kite_client()
    token = _resolve_future_token(kite, z)

    end = datetime.now()
    start = end - timedelta(days=int(d["days"]))
    df = _kite_history(
        kite, token, start, end, z["interval"],
        int(z.get("chunk_days", 150)),  # API caps 60minute at 400d
        continuous=bool(z.get("continuous", True)), oi=True,
    )
    if df.empty:
        raise RuntimeError("Kite returned no candles. Check token/dates/subscription.")
    return df[["open", "high", "low", "close", "volume", "oi"]]


# ───────────────────── Zerodha: fallback when Dhan fails ─────────────────
KITE_INTERVALS = {1: "minute", 5: "5minute", 15: "15minute", 60: "60minute"}
KITE_CHUNK_DAYS = 60  # within Kite's cap for every interval above
KITE_INDEX, KITE_VOLUME_PROXY = "NIFTY 50", "NIFTYBEES"


def _kite_interval(minutes: int) -> str:
    if minutes not in KITE_INTERVALS:
        raise SystemExit(f"Kite has no {minutes}-minute candles, so it cannot replace Dhan here.")
    return KITE_INTERVALS[minutes]


def _kite_nse_tokens(kite, *symbols: str) -> list:
    inst = pd.DataFrame(kite.instruments("NSE")).set_index("tradingsymbol")["instrument_token"]
    return [int(inst.loc[s]) for s in symbols]


def download_kite_underlying(cfg: dict) -> pd.DataFrame:
    """Same series as download_dhan_underlying: Nifty index OHLC weighted by NIFTYBEES volume."""
    kite = _kite_client()
    interval = _kite_interval(int(cfg["dhan_underlying"]["interval"]))
    end = _now_ist()
    start = (end - timedelta(days=int(cfg["data"]["days"]) - 1)).replace(hour=9, minute=0)
    index_token, volume_token = _kite_nse_tokens(kite, KITE_INDEX, KITE_VOLUME_PROXY)

    px = _kite_history(kite, index_token, start, end, interval, KITE_CHUNK_DAYS, KITE_INDEX)
    if px.empty:
        raise RuntimeError("Kite returned no Nifty candles. Check token/dates/subscription.")
    vol = _kite_history(kite, volume_token, start, end, interval, KITE_CHUNK_DAYS, KITE_VOLUME_PROXY)
    px["volume"] = vol["volume"].reindex(px.index) if not vol.empty else float("nan")
    missing = int(px["volume"].isna().sum())
    if missing:
        log.warning(f"{missing} bars had no proxy volume; filled with 0.")
    px["volume"] = px["volume"].fillna(0)
    return px[["open", "high", "low", "close", "volume"]]


def _kite_fill_path(cfg: dict) -> Path:
    dh = cfg["dhan"]
    return Path(dh["output_csv"]).parent / "kite_parts" / f"{_parts_tag(dh)}.csv"


def _dhan_covered_until(dh: dict):
    """Last day that EVERY saved strike/side file reaches, or None when nothing is saved."""
    parts = Path(dh["output_csv"]).parent / "dhan_parts"
    cov_all = _load_coverage(parts)
    ends = []
    for offset in range(int(dh["strike_min"]), int(dh["strike_max"]) + 1):
        for opt_type in ("CALL", "PUT"):
            cov = _part_coverage(parts / f"{_parts_tag(dh)}_{opt_type}_{offset}.csv", cov_all)
            if cov:
                ends.append(cov[1])
    return min(ends) if ends else None


def _fetch_kite_options(kite, cfg: dict, start: datetime, end: datetime) -> pd.DataFrame:
    """Rebuild Dhan's rows for [start, end]: ATM is the index close rounded to the strike step,
    and the contract is the nearest weekly expiry. Only that live expiry can be fetched."""
    dh = cfg["dhan"]
    underlying = cfg.get("zerodha", {}).get("underlying", "NIFTY")
    step = int(dh.get("strike_step", 50))
    interval = _kite_interval(int(dh["interval"]))

    inst = pd.DataFrame(kite.instruments("NFO"))
    inst = inst[(inst["name"] == underlying) & inst["instrument_type"].isin(["CE", "PE"])].copy()
    inst["expiry"] = pd.to_datetime(inst["expiry"]).dt.date
    inst = inst[inst["expiry"] >= _now_ist().date()]
    if inst.empty:
        raise RuntimeError(f"No active {underlying} options found on NFO.")
    expiry = inst["expiry"].min()
    inst = inst[inst["expiry"] == expiry]

    # Days up to the previous weekly expiry belonged to a contract Kite no longer serves.
    first_day = expiry - timedelta(days=6)
    if start.date() < first_day:
        log.warning(f"Kite cannot serve {start.date()} -> {first_day - timedelta(days=1)}: those "
                    "days traded an expired contract. They stay missing from the options file "
                    "until Dhan works again.")
        start = datetime.combine(first_day, start.time())
    if start >= end:
        return pd.DataFrame()

    index_token = _kite_nse_tokens(kite, KITE_INDEX)[0]
    spot = _kite_history(kite, index_token, start, end, interval, KITE_CHUNK_DAYS, KITE_INDEX)
    if spot.empty:
        return pd.DataFrame()
    spot = spot["close"].rename("spot").reset_index()
    atm = (spot["spot"] / step).round() * step
    grid = pd.concat([
        spot.assign(option_type=side, strike_offset=offset, strike=atm + offset * step)
        for side in ("CE", "PE")
        for offset in range(int(dh["strike_min"]), int(dh["strike_max"]) + 1)
    ], ignore_index=True)

    tokens = {(r.instrument_type, float(r.strike)): int(r.instrument_token)
              for r in inst.itertuples()}
    wanted = grid[["option_type", "strike"]].drop_duplicates()
    log.info(f"Kite options: expiry {expiry}, {len(wanted)} contracts, {start.date()} -> {end.date()}")
    frames = []
    for side, strike in wanted.itertuples(index=False):
        token = tokens.get((side, float(strike)))
        if token is None:
            continue
        bars = _kite_history(kite, token, start, end, interval, KITE_CHUNK_DAYS,
                             f"{strike:.0f}{side}", oi=True)
        if not bars.empty:
            frames.append(bars.reset_index().assign(option_type=side, strike=strike))
    if not frames:
        return pd.DataFrame()
    out = grid.merge(pd.concat(frames, ignore_index=True), on=["datetime", "option_type", "strike"])
    out["iv"] = float("nan")  # Kite has no IV
    out["expiry_flag"], out["expiry_code"] = dh["expiry_flag"], int(dh["expiry_code"])
    return out[OPTION_COLS]


def download_kite_options(cfg: dict) -> None:
    """Top up the saved Dhan options with the days after them, then rebuild the options CSV."""
    dh = cfg["dhan"]
    if (dh["expiry_flag"], int(dh["expiry_code"])) != ("WEEK", 1):
        raise SystemExit("The Kite options fallback only supports the nearest weekly expiry "
                         "(expiry_flag WEEK, expiry_code 1).")
    covered = _dhan_covered_until(dh)
    if covered is None:
        raise SystemExit("No saved Dhan options to top up, and Kite does not serve expired "
                         "contracts, so the history needs a valid Dhan token.")
    kite = _kite_client()
    end = _now_ist()
    start = datetime.combine(covered + timedelta(days=1), datetime.min.time()).replace(hour=9)
    df = _fetch_kite_options(kite, cfg, start, end) if start < end else pd.DataFrame()
    if df.empty:
        log.info(f"Kite has no new option bars after {covered}.")
    else:
        fill = _kite_fill_path(cfg)
        fill.parent.mkdir(parents=True, exist_ok=True)
        old = _read_kite_fill(fill)
        if old is not None:
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(OPTION_KEY, keep="last")
        df.to_csv(fill, index=False)
        log.info(f"Saved {len(df)} Kite option rows -> {fill}")
    assemble_options(cfg)


# ─────────────────────────────── yfinance ───────────────────────────────
def _fetch_yf(symbol: str, d: dict) -> pd.DataFrame:
    import yfinance as yf

    log.info(f"Downloading {symbol} | interval={d['interval']} | period={d['period']}")
    raw = yf.Ticker(symbol).history(
        period=d["period"], interval=d["interval"], auto_adjust=False
    )
    if raw.empty:
        raise RuntimeError(f"No data returned by yfinance for {symbol}. Check symbol/period.")

    raw = raw.rename(
        columns={"Open": "open", "High": "high", "Low": "low",
                 "Close": "close", "Volume": "volume"}
    )[["open", "high", "low", "close", "volume"]].copy()

    raw.index = _localize_ist(raw.index)
    raw.index.name = "datetime"
    return raw[~raw.index.duplicated(keep="last")].sort_index()


def download_yfinance(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    raw = _fetch_yf(d["symbol"], d)

    # Volume from a proxy when the primary symbol has none (e.g. the index) --
    vol_sym = d.get("volume_symbol")
    if vol_sym and not (raw["volume"] > 0).any():
        vol = _fetch_yf(vol_sym, d)["volume"]
        raw["volume"] = vol.reindex(raw.index)
        missing = raw["volume"].isna().sum()
        if missing:
            log.warning(f"{missing} bars had no proxy volume; filled with 0.")
        raw["volume"] = raw["volume"].fillna(0)
        log.info(f"Volume taken from {vol_sym}")
    if not (raw["volume"] > 0).any():
        log.warning("Volume is all zero — VWAP will be NaN and no signals will fire.")

    # Optional OI merge ----------------------------------------------------
    oi_csv = d.get("oi_csv")
    if oi_csv and Path(oi_csv).exists():
        log.info(f"Merging OI from {oi_csv}")
        oi_df = pd.read_csv(oi_csv, parse_dates=["datetime"], index_col="datetime")
        oi_df.index = _localize_ist(oi_df.index)
        oi_df = oi_df[~oi_df.index.duplicated(keep="last")].sort_index()
        raw = raw.join(oi_df[["oi"]], how="left")
        raw["oi"] = raw["oi"].ffill()
        if raw["oi"].isna().all():
            log.warning("OI column was merged but is entirely NaN — check your OI CSV.")
    else:
        log.warning("No OI CSV supplied — strategy will treat OI filter as disabled.")
    return raw


# ─────────────────────── Dhan: underlying prices ────────────────────────
INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"


def _decode_epoch(ts: pd.Series, interval: int, lo, hi):
    """Dhan does not document the timestamp convention, so try the plausible ones and keep the
    one that puts the bars on the exchange clock (09:15-15:30 IST, :15 past the hour for hourly
    bars) inside the requested dates."""
    ts = ts.astype("int64")
    cands = {
        "unix epoch (UTC)": lambda x: pd.to_datetime(x, unit="s", utc=True).dt.tz_convert(IST),
        "unix epoch with IST clock": lambda x: pd.to_datetime(x, unit="s").dt.tz_localize(IST),
        "1980 epoch (UTC)": lambda x: pd.to_datetime(x + 315532800, unit="s", utc=True).dt.tz_convert(IST),
        "1980 epoch with IST clock": lambda x: pd.to_datetime(x + 315532800, unit="s").dt.tz_localize(IST),
    }
    best_name, best_t, best_score = None, None, -1.0
    for name, fn in cands.items():
        try:
            t = fn(ts)
        except Exception:
            continue
        mins = t.dt.hour * 60 + t.dt.minute
        score = ((mins >= 9 * 60 + 15) & (mins < 15 * 60 + 30)).mean()
        score *= ((t.dt.date >= lo - timedelta(days=1)) & (t.dt.date <= hi + timedelta(days=1))).mean()
        if interval == 60:
            score *= (t.dt.minute == 15).mean()
        if score > best_score:
            best_name, best_t, best_score = name, t, score
    if best_score < 0.9:
        raise RuntimeError(
            f"Could not work out Dhan's timestamp format (best match {best_name}, score "
            f"{best_score:.2f}). First values: {ts.head(3).tolist()}")
    return best_name, best_t


def _fetch_intraday(session, spec: dict, interval: int, windows: list, label: str):
    frames, fmt = [], None
    for i, (f, t) in enumerate(windows, 1):
        body = {
            "securityId": str(spec["security_id"]),
            "exchangeSegment": spec["exchange_segment"],
            "instrument": spec["instrument"],
            "interval": str(interval),
            "oi": False,
            "fromDate": f"{f} 09:00:00",
            "toDate": f"{t} 15:30:00",
        }
        data = _dhan_request(session, body, url=INTRADAY_URL, key=None)
        time.sleep(0.3)
        if not data or not data.get("timestamp"):
            log.info(f"  {label}: window {i}/{len(windows)} ({f} -> {t}): no data")
            continue
        raw = pd.DataFrame({k: data[k] for k in ("timestamp", "open", "high", "low", "close", "volume")})
        name, when = _decode_epoch(raw["timestamp"], interval, f, t)
        if fmt is None:
            fmt = name
            log.info(f"  {label}: timestamps look like a {fmt}")
        elif name != fmt:
            log.warning(f"  {label}: window {f} decoded as {name}, earlier windows as {fmt}")
        raw.index = when
        frames.append(raw.drop(columns="timestamp"))
        log.info(f"  {label}: window {i}/{len(windows)} ({f} -> {t}): {len(raw)} bars")
    if not frames:
        raise RuntimeError(f"Dhan returned no {label} data. Check the token and security id.")
    out = pd.concat(frames)
    out.index.name = "datetime"
    return out[~out.index.duplicated(keep="last")].sort_index()


def download_dhan_underlying(cfg: dict) -> pd.DataFrame:
    """Nifty index OHLC from Dhan's intraday charts. The index has no volume, so (like the
    yfinance path) the volume of a liquid proxy, the NIFTYBEES ETF, is used to weight VWAP."""
    u = cfg["dhan_underlying"]
    session = _dhan_session()
    interval = int(u["interval"])
    end = _now_ist().date()  # up to the run time; main() drops the bar still forming
    start = end - timedelta(days=int(cfg["data"]["days"]) - 1)
    span = min(int(u.get("chunk_days", 85)), 90)  # Dhan allows at most 90 days per request
    windows, cur = [], start
    while cur <= end:
        to = min(cur + timedelta(days=span - 1), end)
        windows.append((cur, to))
        cur = to + timedelta(days=1)
    log.info(f"Dhan underlying: {interval}-minute bars, {start} -> {end}, {len(windows)} windows per series")

    px = _fetch_intraday(session, u["price"], interval, windows, "Nifty index")
    if u.get("volume"):
        vol = _fetch_intraday(session, u["volume"], interval, windows, "volume proxy")["volume"]
        px["volume"] = vol.reindex(px.index)
        missing = int(px["volume"].isna().sum())
        if missing:
            log.warning(f"{missing} bars had no proxy volume; filled with 0.")
        px["volume"] = px["volume"].fillna(0)
    return px[["open", "high", "low", "close", "volume"]]


# ─────────────────────────── Daily Nifty candles ─────────────────────────
HISTORICAL_URL = "https://api.dhan.co/v2/charts/historical"


def _decode_daily_epoch(ts: pd.Series, lo, hi) -> pd.Series:
    """Daily timestamps as IST midnight. Dhan does not document the epoch, so keep the
    convention (unix or 1980) that lands the dates inside the requested range."""
    ts = ts.astype("int64")
    for offset in (0, 315532800):
        t = pd.to_datetime(ts + offset, unit="s", utc=True).dt.tz_convert(IST).dt.normalize()
        d = t.dt.date
        if ((d >= lo - timedelta(days=1)) & (d <= hi + timedelta(days=1))).mean() >= 0.9:
            return t
    raise RuntimeError(f"Could not work out Dhan's daily timestamp format. First values: {ts.head(3).tolist()}")


def _fetch_daily(session, spec: dict, windows: list, label: str) -> pd.DataFrame:
    frames = []
    for i, (f, t) in enumerate(windows, 1):
        body = {
            "securityId": str(spec["security_id"]),
            "exchangeSegment": spec["exchange_segment"],
            "instrument": spec["instrument"],
            "expiryCode": 0,
            "oi": False,
            "fromDate": f.isoformat(),
            "toDate": (t + timedelta(days=1)).isoformat(),  # toDate is exclusive for daily candles
        }
        data = _dhan_request(session, body, url=HISTORICAL_URL, key=None)
        time.sleep(0.3)
        if not data or not data.get("timestamp"):
            log.info(f"  {label}: window {i}/{len(windows)} ({f} -> {t}): no data")
            continue
        raw = pd.DataFrame({k: data[k] for k in ("timestamp", "open", "high", "low", "close", "volume")})
        raw.index = _decode_daily_epoch(raw["timestamp"], f, t)
        frames.append(raw.drop(columns="timestamp"))
        log.info(f"  {label}: window {i}/{len(windows)} ({f} -> {t}): {len(raw)} bars")
    if not frames:
        raise RuntimeError(f"Dhan returned no daily {label} data. Check the token and security id.")
    out = pd.concat(frames)
    out.index.name = "datetime"
    return out[~out.index.duplicated(keep="last")].sort_index()


def download_daily(cfg: dict) -> pd.DataFrame:
    """Daily Nifty candles for daily.years years, from Dhan (volume of a proxy symbol, as for the
    hourly series) or from yfinance."""
    d = cfg["daily"]
    now = _now_ist()
    end = now.date()
    start = end - timedelta(days=round(float(d["years"]) * 365.25))
    source = d.get("source", "dhan")
    if source == "dhan":
        session = _dhan_session()
        span = max(int(d.get("chunk_days", 365)), 1)
        windows, cur = [], start
        while cur <= end:
            to = min(cur + timedelta(days=span - 1), end)
            windows.append((cur, to))
            cur = to + timedelta(days=1)
        log.info(f"Dhan daily: {start} -> {end}, {len(windows)} windows per series")
        px = _fetch_daily(session, d["price"], windows, "Nifty index")
        if d.get("volume"):
            vol = _fetch_daily(session, d["volume"], windows, "volume proxy")["volume"]
            px["volume"] = vol.reindex(px.index)
            missing = int(px["volume"].isna().sum())
            if missing:
                log.warning(f"{missing} days had no proxy volume; filled with 0.")
            px["volume"] = px["volume"].fillna(0)
        px = px[["open", "high", "low", "close", "volume"]]
    elif source == "yfinance":
        y = {"interval": "1d", "period": f"{int(round(float(d['years'])))}y"}
        px = _fetch_yf(d["symbol"], y)
        if d.get("volume_symbol") and not (px["volume"] > 0).any():
            px["volume"] = _fetch_yf(d["volume_symbol"], y)["volume"].reindex(px.index).fillna(0)
        px.index = px.index.normalize()
    else:
        raise SystemExit(f"daily.source must be 'dhan' or 'yfinance', not {source!r}")
    if now.time() < DAY_DONE:  # today's candle is still forming
        px = px[px.index.date < end]
    px.index.name = "datetime"
    return px


def build_daily_options(cfg: dict) -> None:
    """Daily weekly-option candles built from the saved hourly options file (Dhan's rolling-option
    API has no daily interval). One row per day, option type and absolute strike: open of the first
    bar, high/low of the day, close/OI/IV of the last bar, volume summed. strike_offset is re-based on
    each day's ATM (first bar's spot, rounded to the strike step), because the hourly offsets move
    with spot during the day."""
    o = cfg["daily_options"]
    sources = [Path(p) for p in ([o["source_csv"]] if isinstance(o["source_csv"], str) else o["source_csv"])]
    for src in sources:
        if not src.exists():
            raise SystemExit(f"{src} not found. Download the hourly options first (--only options).")
    df = pd.concat([pd.read_csv(src) for src in sources], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(IST)
    df = df.sort_values("datetime")
    df["day"] = df["datetime"].dt.normalize()
    step = int(o.get("strike_step", 50))
    atm = (df.groupby("day")["spot"].first() / step).round() * step

    # one row per expiry too: the same strike trades in several weekly contracts at once
    g = df.groupby(["day", "expiry_flag", "expiry_code", "option_type", "strike"], sort=True)
    out = g.agg(spot=("spot", "last"), open=("open", "first"), high=("high", "max"),
                low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
                oi=("oi", "last"), iv=("iv", "last")).reset_index()
    out["strike_offset"] = ((out["strike"] - out["day"].map(atm)) / step).round().astype(int)
    out = out[(out["strike_offset"] >= int(o["strike_min"])) & (out["strike_offset"] <= int(o["strike_max"]))]

    # same cut-off as the hourly files: skip today until the session is over
    now = _now_ist()
    if now.time() < DAY_DONE:
        out = out[out["day"].dt.date < now.date()]
    out = out.rename(columns={"day": "datetime"})[OPTION_COLS].sort_values(["expiry_code", *OPTION_KEY])
    dest = Path(o["output_csv"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    log.info(f"Saved {len(out)} daily option rows -> {dest} "
             f"({out['datetime'].min().date()} -> {out['datetime'].max().date()}, "
             f"{out['datetime'].nunique()} days, {out.groupby(['expiry_code', 'option_type', 'strike_offset']).ngroups} series, "
             f"expiry codes {sorted(out['expiry_code'].unique().tolist())})")


def save_daily(cfg: dict) -> None:
    df = download_daily(cfg)
    out = Path(cfg["daily"]["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    log.info(f"Saved {len(df)} daily rows -> {out} ({df.index.min().date()} -> {df.index.max().date()})")


# ───────────────────────────────── Dhan ─────────────────────────────────
DHAN_URL = "https://api.dhan.co/v2/charts/rollingoption"
DHAN_FIELDS = ["open", "high", "low", "close", "volume", "oi", "iv", "strike", "spot"]
OPTION_KEY = ["datetime", "option_type", "strike_offset"]
OPTION_COLS = ["datetime", "option_type", "strike_offset", "strike", "spot",
               "open", "high", "low", "close", "volume", "oi", "iv", "expiry_flag", "expiry_code"]


class DhanAuthError(SystemExit):
    """Dhan token missing or rejected. Exits like before unless the Kite fallback catches it."""


def _strike_label(offset: int) -> str:
    return "ATM" if offset == 0 else f"ATM{offset:+d}"


def _dhan_request(session: requests.Session, body: dict, retries: int = 6,
                  url: str = DHAN_URL, key: str | None = "data") -> dict | None:
    for attempt in range(retries):
        try:
            r = session.post(url, json=body, timeout=60)
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = min(5 * 2 ** attempt, 120)  # DNS / internet blip: wait it out
            log.warning(f"Network error ({type(e).__name__}); retry {attempt + 1}/{retries} "
                        f"in {wait}s")
            time.sleep(wait)
            if attempt == retries - 1:
                raise SystemExit(
                    "Cannot reach api.dhan.co (check internet/DNS/VPN). Finished strikes are "
                    "saved; re-run the same command to resume."
                )
            continue
        if r.status_code == 200:
            return (r.json().get(key) if key else r.json()) or {}
        if r.status_code in (401, 403) or "DH-901" in r.text:
            raise DhanAuthError(
                f"Dhan rejected the credentials ({r.status_code}): {r.text[:200]}\n"
                "Generate a fresh access token in the Dhan web app (tokens expire after 24h) "
                "and set DHAN_ACCESS_TOKEN and DHAN_CLIENT_ID in this same window."
            )
        if r.status_code in (429, 500, 502, 503, 504) or "DH-904" in r.text:
            time.sleep(2 ** attempt)  # rate limited / transient
            continue
        log.warning(f"Dhan {r.status_code} for {body.get('drvOptionType', body.get('securityId'))} "
                    f"{body.get('strike', '')} {body['fromDate']}: {r.text[:200]}")
        return None
    log.warning(f"Dhan gave up after {retries} tries: {body.get('strike', body.get('securityId'))} "
                f"{body['fromDate']}")
    return None


def _dhan_session() -> requests.Session:
    token = os.environ.get("DHAN_ACCESS_TOKEN")
    if not token:
        raise DhanAuthError("Set the DHAN_ACCESS_TOKEN environment variable.")
    token = token.strip().strip('"')  # stray spaces/quotes from copy-paste break auth
    headers = {"access-token": token, "Accept": "application/json",
               "Content-Type": "application/json"}
    if os.environ.get("DHAN_CLIENT_ID"):
        headers["client-id"] = os.environ["DHAN_CLIENT_ID"].strip()
    else:
        log.warning("DHAN_CLIENT_ID is not set; Dhan may reject the token without it.")
    session = requests.Session()
    session.headers.update(headers)
    return session


def _dhan_body(dh: dict, strike: str, opt_type: str, f, t, **override) -> dict:
    body = {
        "exchangeSegment": dh["exchange_segment"],
        "interval": str(dh["interval"]),
        "securityId": int(dh["security_id"]),
        "instrument": dh["instrument"],
        "expiryFlag": dh["expiry_flag"],
        "expiryCode": int(dh["expiry_code"]),
        "strike": strike,
        "drvOptionType": opt_type,
        "requiredData": DHAN_FIELDS,
        "fromDate": f.isoformat(),
        "toDate": t.isoformat(),
    }
    body.update(override)
    return body


def probe_dhan(cfg: dict) -> None:
    """Send request variants to rollingoption and show which one Dhan accepts."""
    dh, session = cfg["dhan"], _dhan_session()
    today = datetime.now().date()
    # Dhan now allows 1 day per request. Use the most recent weekday that is not today.
    hi = today - timedelta(days=1)
    while hi.weekday() >= 5:
        hi -= timedelta(days=1)
    lo = hi - timedelta(days=1)
    base = _dhan_body(dh, "ATM", "CALL", lo, hi)
    base["expiryFlag"], base["expiryCode"] = "WEEK", 1
    sample = {  # Dhan docs example, with an old date range
        "exchangeSegment": "NSE_FNO", "interval": "1", "securityId": 13,
        "instrument": "OPTIDX", "expiryFlag": "MONTH", "expiryCode": 1,
        "strike": "ATM", "drvOptionType": "CALL",
        "requiredData": ["open", "high", "low", "close", "volume"],
        "fromDate": lo.isoformat(), "toDate": hi.isoformat(),
    }
    variants = [
        ("docs sample, 1-day window (MONTH, interval '1')", sample),
        ("docs sample, WEEK", {**sample, "expiryFlag": "WEEK"}),
        ("docs sample, interval '60'", {**sample, "interval": "60"}),
        ("docs sample, interval 60 (int)", {**sample, "interval": 60}),
        ("docs sample, securityId '13' (str)", {**sample, "securityId": "13"}),
        ("docs sample + oi/iv/strike/spot", {**sample, "requiredData": DHAN_FIELDS}),
        ("1 day, WEEK/1, interval '60'", base),
        ("1 day, WEEK/1, interval 60 (int)", {**base, "interval": 60}),
        ("1 day, MONTH/1, interval '60'", {**base, "expiryFlag": "MONTH"}),
        ("1 day, strike ATM+1", {**base, "strike": "ATM+1"}),
        ("fromDate == toDate (same day)",
         {**base, "fromDate": hi.isoformat(), "toDate": hi.isoformat()}),
        ("3-day window (toDate inclusive? 2 more days)",
         {**base, "fromDate": (lo - timedelta(days=2)).isoformat()}),
    ]
    print(f"Probing {DHAN_URL} (1 day: {lo} -> {hi})")
    for label, body in variants:
        r = session.post(DHAN_URL, json=body, timeout=60)
        detail = r.text[:150]
        if r.status_code == 200:
            ce = (r.json().get("data") or {}).get("ce") or {}
            detail = f"{len(ce.get('timestamp') or [])} bars"
        print(f"  {label:52} HTTP {r.status_code} {detail}")
        time.sleep(0.5)


def _parts_tag(dh: dict) -> str:
    return f"{dh['expiry_flag']}{int(dh['expiry_code'])}_{int(dh['interval'])}m"


def _migrate_parts(parts: Path, dh: dict) -> None:
    """Older runs named saved files without the candle size. Those were all 60-minute data,
    so rename them into the new scheme, but only when this run is also 60-minute."""
    if int(dh["interval"]) != 60 or not parts.exists():
        return
    flag, code = dh["expiry_flag"], int(dh["expiry_code"])
    old = re.compile(rf"^(?:{flag}{code}_)?(CALL|PUT)_(-?\d+)\.csv$")  # WEEK1_CALL_-3.csv or CALL_-3.csv
    for f in sorted(parts.glob("*.csv")):
        m = old.match(f.name)
        if m and (f.name.startswith(f"{flag}{code}_") or (flag, code) == ("WEEK", 1)):
            target = parts / f"{_parts_tag(dh)}_{m.group(1)}_{m.group(2)}.csv"
            if not target.exists():
                f.rename(target)
                log.info(f"Renamed saved file {f.name} -> {target.name}")


def _load_coverage(parts: Path) -> dict:
    f = parts / "_coverage.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save_coverage(parts: Path, cov: dict) -> None:
    (parts / "_coverage.json").write_text(json.dumps(cov, indent=1, sort_keys=True))


def _part_coverage(part: Path, cov_all: dict):
    """Date range a saved file is known to cover: from the manifest, or (for files saved
    before the manifest existed) inferred from the data's own first and last day."""
    if part.name in cov_all:
        a, b = cov_all[part.name]
        return datetime.fromisoformat(a).date(), datetime.fromisoformat(b).date()
    if not part.exists() or part.stat().st_size == 0:
        return None
    ts = pd.read_csv(part, usecols=["timestamp"])["timestamp"]
    if ts.empty:
        return None
    d = pd.to_datetime(ts, unit="s", utc=True).dt.tz_convert(IST).dt.date
    return d.min(), d.max()


def download_dhan(cfg: dict) -> pd.DataFrame:
    dh = cfg["dhan"]
    session = _dhan_session()

    # Dhan (Oct 2026): fromDate/toDate are BOTH inclusive and a request may span at
    # most 2 calendar days. Today gets its own window, refetched every run until the
    # session is over, so a run during market hours never marks today as complete.
    now = _now_ist()
    today = end = now.date()
    yesterday = today - timedelta(days=1)
    start = end - timedelta(days=int(cfg["data"]["days"]) - 1)
    span = int(dh.get("chunk_days", 2))
    lo, hi = int(dh["strike_min"]), int(dh["strike_max"])
    windows = []
    cur = start
    while cur <= yesterday:
        to = min(cur + timedelta(days=span - 1), yesterday)
        days = [cur + timedelta(days=i) for i in range((to - cur).days + 1)]
        if any(d.weekday() < 5 for d in days):  # skip weekend-only windows
            windows.append((cur, to))
        cur = to + timedelta(days=1)
    if today.weekday() < 5:
        windows.append((today, today))

    flag, code = dh["expiry_flag"], int(dh["expiry_code"])
    tag = _parts_tag(dh)  # parts are keyed by expiry AND candle size, so changing either never reuses stale files
    parts = Path(dh["output_csv"]).parent / "dhan_parts"  # one file per expiry/interval/strike/side
    parts.mkdir(parents=True, exist_ok=True)
    _migrate_parts(parts, dh)

    # What is already saved? Each file records the dates it covers, so a longer history
    # (or yesterday's data) only fetches the windows that are missing and merges them in.
    cov_all = _load_coverage(parts)
    plan = {}
    for offset in range(lo, hi + 1):
        for opt_type in ("CALL", "PUT"):
            part = parts / f"{tag}_{opt_type}_{offset}.csv"
            cov = _part_coverage(part, cov_all)
            if cov and (cov[1] < start or cov[0] > end):
                cov = None  # saved range does not touch the wanted range: do not trust it
            needed = [w for w in windows if not (cov and w[0] >= cov[0] and w[1] <= cov[1])]
            plan[(offset, opt_type)] = (part, cov, needed)
    total = max(sum(len(v[2]) for v in plan.values()), 1)
    saved_windows = len(plan) * len(windows) - sum(len(v[2]) for v in plan.values())
    est_h = total * (float(dh.get("sleep_sec", 0.25)) + 0.3) / 3600
    log.info(f"Dhan {flag} expiry #{code} | {dh['interval']}-min bars | {start} -> {end}: "
             f"{hi - lo + 1} strikes x 2 types x {len(windows)} windows; "
             f"{saved_windows} windows already saved, {total} requests to make (~{est_h:.1f}h). "
             f"Progress is saved in {parts} after each strike/side.")
    frames, n, failures = [], 0, 0
    pairs_total, pair_no, rows = len(plan), 0, 0
    every = int(dh.get("progress_every", 25))  # log a line every N requests
    t0, fetched = time.monotonic(), 0  # fetched = requests made this run

    def progress() -> str:
        elapsed = time.monotonic() - t0
        eta = (total - n) * elapsed / fetched if fetched else 0
        return (f"{n / total:6.1%} | {n}/{total} requests | "
                f"elapsed {timedelta(seconds=int(elapsed))} | ETA {timedelta(seconds=int(eta))}")

    for (offset, opt_type), (part, cov, needed) in plan.items():
        pair_no += 1
        label = f"[{pair_no}/{pairs_total}] {_strike_label(offset)} {opt_type}"
        existing = pd.read_csv(part) if part.exists() and part.stat().st_size > 0 else None
        if not needed:
            if existing is not None:
                frames.append(existing)
                rows += len(existing)
            log.info(f"{label}: already downloaded, skipped | {progress()}")
            continue
        pair, complete, today_ok = [], True, False
        for w, (f, t) in enumerate(needed, 1):
            n += 1
            fetched += 1
            if w % every == 0:
                log.info(f"{label}: window {w}/{len(needed)} ({f} -> {t}) | {progress()}")
            data = _dhan_request(
                session, _dhan_body(dh, _strike_label(offset), opt_type, f, t))
            if f == today:  # Dhan may not serve the running day yet: not a reason to abort
                if data is None:
                    log.warning(f"{label}: Dhan did not serve today ({today}); will retry next run")
            else:
                failures = failures + 1 if data is None else 0
                complete &= data is not None
            if failures >= 10:
                raise SystemExit(
                    "10 Dhan requests in a row failed, so the request parameters are "
                    "probably wrong (or the dates are older than Dhan serves: lower data.days). "
                    "Run: python download_data.py --probe")
            time.sleep(float(dh.get("sleep_sec", 0.25)))
            side = (data or {}).get("ce" if opt_type == "CALL" else "pe")
            if not side or not side.get("timestamp"):
                continue
            today_ok |= f == today
            df = pd.DataFrame({
                k: side.get(k) or [None] * len(side["timestamp"])
                for k in ["timestamp", *DHAN_FIELDS]
            })
            df["option_type"] = "CE" if opt_type == "CALL" else "PE"
            df["strike_offset"] = offset
            df["expiry_flag"], df["expiry_code"] = flag, code
            pair.append(df)

        got = [x for x in [existing, *pair] if x is not None]
        merged = (pd.concat(got, ignore_index=True).drop_duplicates("timestamp", keep="last")
                  .sort_values("timestamp") if got else None)
        if merged is not None:
            frames.append(merged)
            rows += len(merged)
        if complete:
            if merged is not None:
                merged.to_csv(part, index=False)
            else:
                part.write_text("")  # genuinely empty: do not refetch
            # Today only counts as covered once the session is over and its bars arrived.
            done = today if today_ok and now.time() >= DAY_DONE else yesterday
            lo_d, hi_d = (min(cov[0], start), max(cov[1], done)) if cov else (start, done)
            cov_all[part.name] = [lo_d.isoformat(), hi_d.isoformat()]
            _save_coverage(parts, cov_all)
        else:
            log.warning(f"{label}: some days failed; not checkpointed, will retry on re-run")
        log.info(f"{label}: DONE, {sum(len(d) for d in pair)} new bars | "
                 f"{progress()} | {rows} rows so far")

    return _combine_parts(frames, flag, code)


def _combine_parts(frames: list, flag: str, code: int) -> pd.DataFrame:
    if not frames:
        raise RuntimeError("Dhan returned no option data. Check token/data-API access.")
    out = pd.concat(frames, ignore_index=True)
    out["expiry_flag"], out["expiry_code"] = flag, code  # also fills migrated legacy parts
    out["datetime"] = pd.to_datetime(out.pop("timestamp"), unit="s", utc=True).dt.tz_convert(IST)
    out = out.drop_duplicates(OPTION_KEY, keep="last")
    return out[OPTION_COLS].sort_values(OPTION_KEY)


def assemble_options(cfg: dict) -> None:
    """Build dhan.output_csv from the saved parts only (no network, no token)."""
    dh = cfg["dhan"]
    flag, code = dh["expiry_flag"], int(dh["expiry_code"])
    parts = Path(dh["output_csv"]).parent / "dhan_parts"
    _migrate_parts(parts, dh)
    files = sorted(parts.glob(f"{_parts_tag(dh)}_*.csv"))
    frames = [pd.read_csv(f) for f in files if f.stat().st_size > 0]
    log.info(f"Assembling {len(frames)} saved strike/side files from {parts}")
    offsets = sorted({int(o) for fr in frames for o in fr["strike_offset"].unique()})
    log.info(f"Strike offsets present: {offsets[0]} .. {offsets[-1]} "
             f"({len(offsets)} of {int(dh['strike_max']) - int(dh['strike_min']) + 1} configured)")
    _save_options(_combine_parts(frames, flag, code), cfg)


def _kite_fallback_on(cfg: dict) -> bool:
    return bool(cfg.get("dhan", {}).get("kite_fallback", True))


def download_options(cfg: dict) -> None:
    try:
        df = download_dhan(cfg)
    except DhanAuthError as e:
        if not _kite_fallback_on(cfg):
            raise
        log.warning(f"{e}\nFalling back to Kite for the options.")
        download_kite_options(cfg)
        return
    _save_options(df, cfg)


def _read_kite_fill(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(IST)
    return df


def _with_kite_fill(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add saved Kite rows for the days Dhan has not covered yet. Dhan wins on any overlap."""
    kite = _read_kite_fill(_kite_fill_path(cfg))
    if kite is None:
        return df
    covered = _dhan_covered_until(cfg["dhan"])
    if covered is not None:
        kite = kite[kite["datetime"].dt.date > covered]
    if kite.empty:
        return df
    log.info(f"Adding {len(kite)} Kite option rows after the Dhan data (no IV on these)")
    out = pd.concat([kite[OPTION_COLS], df], ignore_index=True)
    return out.drop_duplicates(OPTION_KEY, keep="last").sort_values(OPTION_KEY)


def _save_options(df: pd.DataFrame, cfg: dict) -> None:
    df = _with_kite_fill(df, cfg)
    df = df[_closed(df["datetime"], int(cfg["dhan"]["interval"]))]
    out = Path(cfg["dhan"]["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info(f"Saved {len(df)} option rows -> {out}")
    log.info(f"Date range {df['datetime'].min()} -> {df['datetime'].max()} | "
             f"distinct strikes {df['strike'].nunique()} | "
             f"bars/day {len(df) / df['datetime'].dt.date.nunique() / df.groupby(['option_type', 'strike_offset']).ngroups:.1f} | "
             f"OI coverage {df['oi'].notna().mean():.1%}")


# ──────────────────────────────── Main ──────────────────────────────────
def main(config_path: str, only: str | None = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if only == "daily":
        save_daily(cfg)
        return
    if only == "daily_options":
        build_daily_options(cfg)
        return

    if only in (None, "underlying"):
        source = cfg["data"].get("source", "yfinance")
        if source == "dhan":
            try:
                df = download_dhan_underlying(cfg)
            except DhanAuthError as e:
                if not _kite_fallback_on(cfg):
                    raise
                log.warning(f"{e}\nFalling back to Kite for the underlying.")
                df = download_kite_underlying(cfg)
        else:
            df = download_zerodha(cfg) if source == "zerodha" else download_yfinance(cfg)
        df.index.name = "datetime"
        minutes = _bar_minutes(cfg, source)
        if minutes:
            df = df[_closed(df.index.to_series(), minutes).to_numpy()]

        out = Path(cfg["data"]["output_csv"])
        out.parent.mkdir(parents=True, exist_ok=True)
        backup = out.with_name(f"{out.stem}_previous{out.suffix}")
        if out.exists() and not backup.exists():  # keep the file this one replaces, once
            shutil.copy2(out, backup)
            log.info(f"Kept the previous {out.name} as {backup.name}")
        df.to_csv(out)
        log.info(f"Saved {len(df)} rows -> {out}")
        if "oi" in df.columns:
            log.info(f"OI coverage: {df['oi'].notna().mean():.1%} of bars")

    if only == "options" or (only is None and cfg.get("dhan", {}).get("enabled")):
        download_options(cfg)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--login", action="store_true",
                   help="Generate today's Kite access token and exit")
    p.add_argument("--only", choices=["underlying", "options", "daily", "daily_options"],
                   help="Download just one dataset (daily = daily Nifty candles; daily_options = daily "
                        "option candles built from the hourly options file; see the daily config sections)")
    p.add_argument("--assemble", action="store_true",
                   help="Build the options CSV from saved dhan_parts (no download)")
    p.add_argument("--probe", action="store_true",
                   help="Test Dhan expiry flag/code combinations and exit")
    a = p.parse_args()
    if a.login:
        kite_login()
    elif a.assemble:
        assemble_options(yaml.safe_load(open(a.config, encoding="utf8")))
    elif a.probe:
        probe_dhan(yaml.safe_load(open(a.config, encoding="utf8")))
    else:
        main(a.config, a.only)
