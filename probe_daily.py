"""Throwaway diagnostic: which request shape does Dhan's daily /charts/historical accept today?
Run in the terminal where DHAN_ACCESS_TOKEN / DHAN_CLIENT_ID are set:  python probe_daily.py"""
from datetime import date, timedelta
import download_data as d

s = d._dhan_session()
end = date.today()


def t(label, days, **over):
    body = {"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX", "expiryCode": 0,
            "oi": False, "fromDate": (end - timedelta(days=days)).isoformat(),
            "toDate": end.isoformat()}
    body.update(over)
    r = s.post(d.HISTORICAL_URL, json=body, timeout=60)
    n = len(r.json().get("timestamp", [])) if r.status_code == 200 else "-"
    print(f"{label:34} HTTP {r.status_code}  bars={n}  {'' if r.status_code == 200 else r.text[:110]}")


for n in (5, 30, 90, 180, 365):
    t(f"window {n}d", n)
t("30d, no expiryCode/oi", 30, expiryCode=None, oi=None)
t("30d, instrument INDEX, oi omitted", 30, oi=None)

print("--- more variants (30d window) ---")
t("securityId as int", 30, securityId=13)
t("dates with time", 30, fromDate=(end - timedelta(days=30)).isoformat() + " 09:15:00",
  toDate=end.isoformat() + " 15:30:00")
t("NIFTYBEES equity", 30, securityId="10576", exchangeSegment="NSE_EQ", instrument="EQUITY")
t("only required fields", 30, expiryCode=None, oi=None)
t("lowercase-free: exchangeSegment IDX_I str", 30, exchangeSegment="IDX_I", instrument="INDEX", expiryCode=0, oi=False)

print("--- intraday endpoint, same index (does it still work?) ---")
b = {"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX", "interval": "60",
     "oi": False, "fromDate": (end - timedelta(days=5)).isoformat() + " 09:15:00",
     "toDate": end.isoformat() + " 15:30:00"}
r = s.post(d.INTRADAY_URL, json=b, timeout=60)
print("intraday", r.status_code, r.text[:160])
print("client-id header set:", "client-id" in s.headers)
