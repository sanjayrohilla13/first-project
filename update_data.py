"""
One command to top up every dataset the strategies read (hourly and daily).

    python update_data.py                    # everything, in dependency order
    python update_data.py --only hourly      # hourly underlying + options (expiry 1 and 2)
    python update_data.py --only daily       # daily candles + daily options (needs the hourly options files)
    python update_data.py --skip-expiry2     # skip the next-weekly options (also skips daily options)
    python update_data.py --futures          # also top up Nifty futures from Kite (run --login first)
    python update_data.py --only minute      # 1-minute index + options into data_minute/ (add --futures for futures)
    python update_data.py --minute           # everything above plus the 1-minute files
    python update_data.py --dry-run          # just print the steps

Each step calls download_data.py with the matching download config; the downloader resumes from
what is already saved, so re-running only fetches the days after the last saved bar.
Needs DHAN_ACCESS_TOKEN (and DHAN_CLIENT_ID) in the environment (the token expires every 24h).
A failed step does not stop the rest, except that daily options are skipped if the hourly
options they are built from failed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DL = str(HERE / "download_data.py")

HOURLY = "config_download_5y_60m.yaml"
HOURLY2 = "config_download_5y_60m_expiry2.yaml"
DAILY = "config_download_5y_daily.yaml"
FUT = "config_download_5y_futures.yaml"
MINUTE = "config_download_minute.yaml"

# (key, group, config, --only, label, depends_on keys)
STEPS = [
    ("h_under", "hourly", HOURLY, "underlying", "Hourly Nifty prices", []),
    ("h_opt1", "hourly", HOURLY, "options", "Hourly options, nearest weekly (expiry 1)", []),
    ("h_opt2", "hourly", HOURLY2, "options", "Hourly options, next weekly (expiry 2)", []),
    ("fut", "futures", FUT, "underlying", "Hourly Nifty futures (Kite)", []),
    ("m_under", "minute", MINUTE, "underlying", "1-minute Nifty prices", []),
    ("m_opt", "minute", MINUTE, "options", "1-minute options, nearest weekly", []),
    ("m_fut", "minute_futures", MINUTE, "futures", "1-minute Nifty futures (Kite)", []),
    ("d_under", "daily", DAILY, "daily", "Daily Nifty candles", []),
    ("d_opt", "daily", DAILY, "daily_options", "Daily options (built from hourly files)", ["h_opt1", "h_opt2"]),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["hourly", "daily", "minute"], help="Update just one time frame")
    p.add_argument("--skip-expiry2", action="store_true",
                   help="Skip the next-weekly options (daily options need them, so they are skipped too)")
    p.add_argument("--futures", action="store_true", help="Also update the Kite futures files (needs a Kite login)")
    p.add_argument("--minute", action="store_true",
                   help="Also update the 1-minute files (index, options; futures with --futures). Heavy, so off by default")
    p.add_argument("--dry-run", action="store_true", help="Print the steps without running them")
    a = p.parse_args()

    steps = []
    for key, group, cfg, only, label, deps in STEPS:
        if group in ("futures", "minute_futures"):
            if not a.futures or (group == "minute_futures" and not (a.minute or a.only == "minute")):
                continue
            if a.only and a.only != ("minute" if group == "minute_futures" else "hourly"):
                continue
        elif group == "minute":
            if not (a.minute or a.only == "minute"):
                continue
        elif a.only and group != a.only:
            continue
        if a.skip_expiry2 and key in ("h_opt2", "d_opt"):
            continue
        steps.append((key, cfg, only, label, deps))

    if not a.dry_run and not os.environ.get("DHAN_ACCESS_TOKEN"):
        print("Warning: DHAN_ACCESS_TOKEN is not set in this terminal; Dhan steps will fail "
              "(or fall back to Kite where the config allows).\n")

    results: dict[str, str] = {}
    for key, cfg, only, label, deps in steps:
        cmd = [sys.executable, DL, "--config", str(HERE / "configurations" / cfg), "--only", only]
        print(f"\n=== {label} ===\n$ python download_data.py --config configurations/{cfg} --only {only}", flush=True)
        if a.dry_run:
            continue
        bad = [d for d in deps if results.get(d) == "FAILED"]
        if bad:
            results[key] = "SKIPPED"
            print(f"Skipped: depends on failed step(s) {', '.join(bad)}")
            continue
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=HERE)
        results[key] = "ok" if rc == 0 else "FAILED"
        print(f"-> {results[key]} ({time.time() - t0:.0f}s)")

    if a.dry_run:
        return 0
    print("\n=== Summary ===")
    for key, cfg, only, label, deps in steps:
        print(f"  {results[key]:8} {label}")
    return 0 if all(v == "ok" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
