@echo off
REM Unattended (batch) run of update_data.py with a timestamped log file.
REM
REM   run_updates.bat                      hourly + daily top-up
REM   run_updates.bat --only minute        1-minute index + options
REM   run_updates.bat --minute --futures   everything incl. 1-minute and Kite futures
REM
REM Any arguments are passed straight to update_data.py. Output goes to logs\update_<date_time>.log
REM and the exit code is 0 only if every step succeeded (so Task Scheduler / scripts can react).
REM
REM Needs DHAN_ACCESS_TOKEN / DHAN_CLIENT_ID (and KITE_* for futures) saved for the user with setx.
REM The Dhan token expires every 24h, so refresh it before a scheduled run.
REM
REM Schedule it weekdays after the close (example, 16:00):
REM   schtasks /Create /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 16:00 /TN "NiftyDataUpdate" /TR "C:\Trading-Application\first-project\run_updates.bat"

cd /d "%~dp0"
if not exist logs mkdir logs

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%i
set LOG=logs\update_%STAMP%.log

echo Started %date% %time% - args: %* > "%LOG%"
python -u update_data.py %* >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo Finished %date% %time% - exit code %RC% >> "%LOG%"

echo Log: %LOG% (exit code %RC%)
exit /b %RC%
