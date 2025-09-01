@echo off
setlocal enabledelayedexpansion

:: Load variables from .env
for /f "usebackq tokens=1* delims==" %%a in (".env") do (
    set "%%a=%%b"
)

:: Remove quotes and normalize sys_path (in case it ends with \)
set "base=!sys_path!"
if "!base:~-1!"=="\" set "base=!base:~0,-1!"

:: Build full script paths
set "collector=!base!\src\data_collectors\tradingview_collector.py"
set "alert=!base!\src\analysis\stats_sport_bet\alert_scan_match.py"

echo "!collector!"
echo "!alert!"

:: Run both scripts in parallel (new windows)
start cmd /k python "!collector!"
start cmd /k python "!alert!"

endlocal
