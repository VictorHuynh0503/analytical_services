@echo off
:: Run tradingview_collector.py
python "D:\Projects\analytical_services\src\analysis\stats_sport_bet\stats_last_5_perf.py"

:: Run alert_scan_match.py
python "D:\Projects\analytical_services\src\analysis\stats_sport_bet\alert_scan_match.py"

pause
