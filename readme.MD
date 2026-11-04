# 0) Kite credentials (first time, or to update them). setx saves for new terminals; $env: applies to this one
setx KITE_API_KEY "your_api_key"
setx KITE_API_SECRET "your_api_secret"
$env:KITE_API_KEY = "your_api_key"
$env:KITE_API_SECRET = "your_api_secret"
python download_data.py --login

# 0b) Dhan token (expires every 24h, so update it daily before downloading)
#     Get it at web.dhan.co > My Profile > Access DhanHQ APIs; the Client ID is shown there too.
#     setx saves for new terminals; $env: applies to this one. Paste the token without quotes or spaces.
#     If the token is missing or expired, the download falls back to Kite (step 0) automatically.
setx DHAN_ACCESS_TOKEN "your_access_token"
setx DHAN_CLIENT_ID "your_client_id"
$env:DHAN_ACCESS_TOKEN = "your_access_token"
$env:DHAN_CLIENT_ID = "your_client_id"
python download_data.py --config config_download_5y_60m.yaml --probe   # optional: HTTP 200 lines = token works

# 1) Fetch hourly data
python download_data.py --config config.yaml

#    Long history (~5 years, hourly). Underlying is quick; options take ~3 hours and resume if stopped
python download_data.py --config config_download_5y_60m.yaml --only underlying
python download_data.py --config config_download_5y_60m.yaml --only options

# 1b) Fetch DAILY data (5 years) into data_daily/, separate from the hourly data in data/
#     Underlying daily candles (Dhan needs the token from 0b; set daily.source: "yfinance" for no token)
python download_data.py --config config_download_5y_daily.yaml --only daily
#     Daily weekly-expiry option candles for BOTH the nearest (expiry_code 1) and next (expiry_code 2)
#     weekly expiry. Dhan has no daily interval for options, so they are built from hourly files:
#       - expiry 1 comes from the hourly options download in step 1 (data/nifty_options_1h.csv)
#       - expiry 2 needs its own hourly download (ATM-3..ATM+3 only, ~2 hours, resumes if stopped):
python download_data.py --config config_download_5y_60m_expiry2.yaml --only options
#     ...then build the daily file from both (nothing is requested from Dhan; re-run after refreshing)
python download_data.py --config config_download_5y_daily.yaml --only daily_options
#     The backtest trades one expiry, set by dhan.expiry_code in config_daily.yaml (default 1)

#     Run a long download in the background (PowerShell, from the project folder). Progress goes to the log.
Start-Process python -ArgumentList "download_data.py --config config_download_5y_60m_expiry2.yaml --only options" `
  -WorkingDirectory (Get-Location) -WindowStyle Hidden `
  -RedirectStandardOutput results\download_exp2.out.log -RedirectStandardError results\download_exp2.log
#     Check it:  is it running? / which one? / progress (Ctrl+C stops watching, not the download)
Get-Process python -ErrorAction SilentlyContinue
Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Select-Object ProcessId, CommandLine
Get-Content results\download_exp2.log -Tail 20 -Wait
#     Stop it (finished strikes are saved; re-run the Start-Process line to resume). This stops ALL python
#     processes; use  Stop-Process -Id <ProcessId>  to stop just one.
Stop-Process -Name python
#     It finishes with a "Saved N option rows" line. Sleep/log-off stops it; just start it again to resume.
#     Settings (years, source, folders, strike range) are in config_download_5y_daily.yaml
#     Backtest on the daily files (results go to results/daily/, apart from the hourly results)
python strategy.py --config config_daily.yaml
#     Same daily data with the CMO + OI strategy (daily version of config_cmo_oi.yaml)
python strategy.py --config config_cmo_oi_daily.yaml

# 2) Single-run backtest (baseline). The HTML report shows CAGR, per-leg entry/exit prices and
#    direction / OI bias / CMO at entry and exit for every trade
python strategy.py --config config.yaml

# 3) Grid search over ADX period/threshold, RSI period, ATR multiplier
python strategy.py --config config.yaml --optimize