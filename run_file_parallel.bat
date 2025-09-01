@echo off
:: Run tradingview_collector.py in new window
start cmd /k python "D:\Projects\analytical_services\src\analysis\stats_sport_bet\stats_last_5_perf.py"

:: Run alert_scan_match.py in new window
start cmd /k python "D:\Projects\analytical_services\src\analysis\stats_sport_bet\alert_scan_match.py"
